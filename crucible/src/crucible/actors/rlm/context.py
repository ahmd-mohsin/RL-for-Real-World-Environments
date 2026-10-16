"""Context management for the very-long-horizon RLM harness (M16, design §3.2; requirements R2, R3, R4, R16, H3, H15).

The prompt (L1) is bounded here; variables (L2) survive in the kernel and every message and full cell output (L3) is in
the ledger, so a compaction rewrites the prompt and loses nothing. This module is pure: it never calls the kernel or the
ledger, and the only model call it makes is the summary request inside ``Compactor.run``.

- ``ContextSpec`` / ``threshold()``: Prime Agent's ceiling ``min(0.95·W, W − max_out − reserve)`` floored at nano-rlm's
  ``W // 2``, then ``≤ seq_len − max_out`` so no segment ever reaches a trainer's silent tail cut (prime-rl
  ``orchestrator/batch.py:385-406``; its RLM configs compact at 98,304 under a 131,072 ``seq_len``).
- ``ContextMeter``: usage-anchored accounting (PA ``session_engine/compaction.rs:39-47,199-295``; nano-rlm
  ``engine.py:1133-1146``). The stress test (03-crucible-stress.md B1) measured 1.42 real tokens per chars/4 token on
  RLM cells; a chars/4 trigger let every segment reach 145-148k real tokens under a 128k budget.
- ``is_overflow`` / ``is_silent_overflow``: the nine provider markers on a 400/413 (shared with ``actors.llm``) plus PA's
  silent rules (``pa-ai/src/utils_inner/overflow.rs:8-98``).
- ``summary_view`` / ``hollow_middle`` / ``Compactor``: nano-rlm's summary loop (``engine.py:1195-1365``,
  ``compaction.py:175-223``, MIT): bounded attempts, an overflowing request hollowed and retried, empty / tool-calling /
  reasoning-only replies resampled, ``compaction_failed`` as a clean typed stop. Unlike PA (``compact_session.rs:63-68``)
  the summarizer sees tool output, capped head + tail.
- ``SUMMARY_PROMPT`` / ``UPDATE_PROMPT``: MiMo's nine-section schema (mimoagent-osr ``compaction/prompt.py:32-115``, MIT),
  adapted (H3), with PA's update mode (``compaction.rs:436-460``) and nano-rlm's REPL note (``compaction.py:24-31``).
- ``rebuild``: system, the task verbatim (MiMo ``compaction/surgery.py:47-77``), the summary with PA's framing
  (``session_engine/messages.rs:12``), a capped ``[python-state]`` row, a boundary marker (MiMo ``surgery.py:108-132``),
  the recent tail cut only before an assistant row (PA ``compaction.rs:297-432``). No fabricated assistant turn.
- ``footer`` / ``wrap_up_notice``: MiMo's ``<context_usage>`` footer from 80 % (``agents/cc/cc_agent.py:72,227-244``) with
  H15's step, turn, budget and elapsed fields.

Every number names its source: ``usage`` (the provider's count) or ``estimate`` (chars/4, calibrated).
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from crucible.actors.llm import CONTEXT_OVERFLOW_MARKERS, ContextOverflow
from crucible.actors.policy import approx_tokens

__all__ = [
    "BOUNDARY_PREFIX", "STATE_PREFIX", "SUMMARY_PREFIX", "SUMMARY_PROMPT", "UPDATE_PROMPT", "CompactionOutcome", "Compactor", "ContextMeter",
    "ContextSpec", "boundary_marker", "footer", "hollow_middle", "is_overflow", "is_silent_overflow", "python_state_row", "rebuild",
    "split_tail", "summary_request", "summary_view", "wrap_up_notice",
]

MODES = ("auto", "action", "none")


# ---- the spec and its threshold ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ContextSpec:
    """How one agent's prompt is bounded. ``mode``: ``auto`` (the harness compacts at ``threshold()``), ``action`` (the
    policy calls ``rlm.compact()``; the hard threshold still holds, R13), ``none`` (an overflow is the scored stop
    ``context_overflow``)."""

    window_tokens: int = 128_000
    max_out: int = 2_000               # the turn's max_tokens
    seq_len: int | None = None         # the trainer's sequence length: threshold ≤ seq_len − max_out
    reserve: int = 16_384              # nano-rlm compaction.py:41, PA compaction.rs:8
    tail_tokens: int = 20_000          # recent tail kept verbatim after a compaction (PA compaction.rs:9); 0 = none
    summary_max_tokens: int | None = None   # None: clamp(window // 64, 2_000, 16_000) (H3: 2k at 128k, ~16k at 1M)
    max_compaction_attempts: int = 5   # nano-rlm config.py:87
    max_compactions: int | None = None  # None = unbounded (but counted)
    mode: str = "auto"
    footer_from: float = 0.8           # MiMo cc_agent.py:72: usage fraction from which observations carry the footer
    tool_chars: int = 2_000            # per tool output in the summarizer's view (head) ...
    tool_tail: int = 500               # ... plus this many tail chars (R4)

    def __post_init__(self) -> None:
        if self.window_tokens <= 0 or self.max_out <= 0 or self.max_out >= self.window_tokens:
            raise ValueError(f"ContextSpec: need 0 < max_out < window_tokens (got max_out={self.max_out}, window_tokens={self.window_tokens})")
        if self.seq_len is not None and self.seq_len <= self.max_out:
            raise ValueError(f"ContextSpec: seq_len ({self.seq_len}) must exceed max_out ({self.max_out})")
        if self.mode not in MODES:
            raise ValueError(f"ContextSpec.mode must be one of {MODES}, got {self.mode!r}")
        if self.max_compaction_attempts < 1:
            raise ValueError("ContextSpec.max_compaction_attempts must be >= 1")
        if self.reserve < 0 or self.tail_tokens < 0 or self.tool_chars < 0 or self.tool_tail < 0:
            raise ValueError("ContextSpec: reserve, tail_tokens, tool_chars and tool_tail must be >= 0")

    def threshold(self) -> int:
        """Compact when the context reaches this many tokens: ``max(W//2, min(int(0.95·W), W − max_out − reserve))`` (PA
        ``compaction.rs:152-163`` with nano-rlm's floor, ``compaction.py:90-92``), then ``≤ seq_len − max_out``. The floor
        never lets a reply overrun the window: the result is also ``≤ W − max_out``.

        Room for the summary call: the compaction request never carries the raw messages. Its input is ``summary_view``
        (tool outputs capped head + tail), hollowed by ``Compactor`` to ``summary_input_tokens()`` = ``W −
        summary_tokens() − reserve`` (and ``≤ seq_len − summary_tokens()``) minus the prompt, so the summary call fits
        the window whatever the context held when it fired (B1: the old compaction call was the segment's largest prompt).
        """
        w = self.window_tokens
        t = max(w // 2, min(int(0.95 * w), w - self.max_out - self.reserve))
        t = min(t, w - self.max_out)
        if self.seq_len is not None:
            t = min(t, self.seq_len - self.max_out)
        return max(1, t)

    def summary_tokens(self) -> int:
        """The summary request's ``max_tokens``: ``summary_max_tokens`` or ``clamp(W // 64, 2_000, 16_000)``, never more than
        a quarter of the window."""
        n = self.summary_max_tokens if self.summary_max_tokens is not None else min(16_000, max(2_000, self.window_tokens // 64))
        return max(1, min(n, self.window_tokens // 4))

    def summary_input_tokens(self) -> int:
        """The most the summary request's input (view + prompt) may hold: ``W − summary_tokens − reserve`` (half of ``W −
        summary_tokens`` on windows too small for the reserve), and ``≤ seq_len − summary_tokens`` so an accepted summary
        is a trainable sample."""
        w, s = self.window_tokens, self.summary_tokens()
        n = max((w - s) // 2, w - s - self.reserve)
        if self.seq_len is not None:
            n = min(n, self.seq_len - s)
        return max(1, n)


# ---- accounting -------------------------------------------------------------------------------------------------------
def _usage_tokens(usage: dict[str, Any] | None) -> tuple[int, int] | None:
    """(input, output) from a usage dict; None when the provider gave no usable count. Input = ``prompt_tokens`` plus
    Anthropic-style cache fields when they are reported apart from it (PA ``compaction.rs:39-47``); OpenAI's
    ``cached_tokens`` is already inside ``prompt_tokens``."""
    u = usage or {}
    p = u.get("prompt_tokens", u.get("input_tokens"))
    if not isinstance(p, (int, float)) or isinstance(p, bool) or p <= 0:
        return None
    p = int(p) + int(u.get("cache_read_input_tokens") or 0) + int(u.get("cache_creation_input_tokens") or 0)
    c = u.get("completion_tokens", u.get("output_tokens"))
    return p, int(c) if isinstance(c, (int, float)) and not isinstance(c, bool) else 0


def _message_text(m: dict[str, Any]) -> str:
    """What a message puts in the prompt, for a tokenizer: content text, tool calls, reasoning."""
    c = m.get("content")
    parts: list[str] = []
    if isinstance(c, str):
        parts.append(c)
    elif isinstance(c, list):
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text") or ""))
            elif isinstance(p, dict) and p.get("type") in ("image_url", "input_audio"):
                continue  # counted by approx_tokens' media estimate
            else:
                parts.append(json.dumps(p, default=str))
    elif c:
        parts.append(json.dumps(c, default=str))
    if m.get("tool_calls"):
        parts.append(json.dumps(m["tool_calls"], default=str))
    for k in ("reasoning_content", "reasoning"):
        if isinstance(m.get(k), str):
            parts.append(m[k])
    return "\n".join(parts)


def _media_tokens(m: dict[str, Any]) -> int:
    c = m.get("content")
    if not isinstance(c, list):
        return 0
    media = [p for p in c if isinstance(p, dict) and p.get("type") in ("image_url", "input_audio")]
    return approx_tokens([{"role": m.get("role"), "content": media}]) - 4 if media else 0


class ContextMeter:
    """Usage-anchored context accounting (R2; PA ``session_engine/compaction.rs:199-295``, nano-rlm ``engine.py:1133-1146``).

    ``anchor(usage, n)`` after every *accepted* model call: the provider's ``prompt + completion`` covers the first ``n``
    messages (pass ``n = len(messages)`` after the reply is appended, so the completion is not counted twice).
    ``tokens(messages)`` = that anchor + an estimate of ``messages[n:]``. ``reset()`` after a compaction drops the anchor,
    so pre-compaction usage never re-triggers (PA ``threshold_compaction_due``); until the next call the size is an
    estimate.

    The estimate for the unanchored suffix is ``approx_tokens`` (chars/4) scaled by the calibration ``ratio`` (real tokens
    per estimated token, learnt at each anchor, never below 1.0) so content that tokenizes densely (JSON-escaped code with
    digits: 1.42x, B1) is not undercounted; with ``tokenizer`` (a callable text -> token count) the suffix is counted
    exactly instead. ``source`` says which kind of number the last ``tokens()`` returned: ``usage`` or ``estimate``."""

    def __init__(self, *, tokenizer: Callable[[str], int] | None = None, per_message: int = 4):
        self.tokenizer = tokenizer
        self.per_message = per_message  # template tokens per message (role markers) when a tokenizer counts the text
        self._anchor: tuple[int, int] | None = None  # (messages covered, prompt + completion tokens)
        self._ratio_pending = False
        self._ratio_sum = 0.0  # running sums for the calibration ratio
        self._ratio_n = 0
        self.ratio = 1.0
        self.source = "estimate"
        self.anchors = 0

    def anchor(self, usage: dict[str, Any] | None, n_messages: int) -> None:
        """After an accepted call: the provider's prompt + completion tokens cover ``messages[:n_messages]``. A blank usage
        (tolerant mode) leaves the meter as it was."""
        got = _usage_tokens(usage)
        if got is None:
            return
        self._anchor = (int(n_messages), got[0] + got[1])
        self._ratio_pending = True
        self.anchors += 1

    def reset(self) -> None:
        """After a compaction: the old anchor described a context that no longer exists. The calibration ratio stays."""
        self._anchor = None
        self._ratio_pending = False
        self.source = "estimate"

    def _estimate(self, messages: list[dict[str, Any]]) -> int:
        if not messages:
            return 0
        if self.tokenizer is not None:
            return sum(int(self.tokenizer(_message_text(m))) + self.per_message + _media_tokens(m) for m in messages)
        return int(approx_tokens(messages) * max(1.0, self.ratio) + 0.999)

    def tokens(self, messages: list[dict[str, Any]]) -> int:
        """The context size of ``messages``: anchor + estimated suffix (``source="usage"``), else a whole estimate."""
        if self._anchor is not None and self._anchor[0] <= len(messages):
            n, tok = self._anchor
            if self._ratio_pending:  # learn the calibration once per anchor (O(context), as often as the anchor changes)
                self._ratio_pending = False
                est = approx_tokens(messages[:n])
                if est > 0:
                    self._ratio_sum += tok / est
                    self._ratio_n += 1
                    self.ratio = self._ratio_sum / self._ratio_n
            self.source = "usage"
            return tok + self._estimate(messages[n:])
        self.source = "estimate"
        return self._estimate(messages)

    def state_dict(self) -> dict[str, Any]:
        return {"anchor": list(self._anchor) if self._anchor else None, "ratio_pending": self._ratio_pending, "ratio_sum": self._ratio_sum,
                "ratio_n": self._ratio_n, "ratio": self.ratio, "source": self.source, "anchors": self.anchors}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        a = d.get("anchor")
        self._anchor = (int(a[0]), int(a[1])) if a else None
        self._ratio_pending = bool(d.get("ratio_pending", False))
        self._ratio_sum, self._ratio_n = float(d.get("ratio_sum", 0.0)), int(d.get("ratio_n", 0))
        self.ratio, self.source, self.anchors = float(d.get("ratio", 1.0)), str(d.get("source", "estimate")), int(d.get("anchors", 0))


# ---- overflow detection -----------------------------------------------------------------------------------------------
# PA pa-ai/src/utils_inner/overflow.rs:41-53: rate limits and outages that can quote a size are never overflows
_NOT_OVERFLOW = re.compile(r"rate limit|too many requests|^(throttling error|service unavailable):", re.IGNORECASE | re.MULTILINE)


def _status(e: BaseException) -> int | None:
    s = getattr(e, "status_code", None)
    if s is None:
        s = getattr(getattr(e, "response", None), "status_code", None)
    return s if isinstance(s, int) else None


def is_overflow(exc: BaseException | None = None, *, status: int | None = None, text: str = "") -> bool:
    """True when a failed request was refused because the prompt does not fit: Crucible's ``ContextOverflow`` (the client
    already classifies 400/413 bodies, 200 error bodies and ``length`` with 0 output), or a 400/413 whose text carries one
    of the nine provider markers (``actors.llm.CONTEXT_OVERFLOW_MARKERS``, ported from nano-rlm ``compaction.py:53-87``).
    Walks ``__cause__``, so a wrapped transport error is still seen. Rate-limit wordings never count (PA overflow.rs:41-53)."""
    details = [text or ""]
    st = status
    seen: set[int] = set()
    e = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, ContextOverflow):
            return True
        if st is None:
            st = _status(e)
        details.append(str(e))
        body = getattr(e, "body", None)
        if body:
            details.append(json.dumps(body, default=str) if not isinstance(body, str) else body)
        resp = getattr(e, "response", None)
        if resp is not None:
            with contextlib.suppress(Exception):  # an unread streaming body: the status and the message still decide
                details.append(str(getattr(resp, "text", "") or ""))
        e = e.__cause__
    d = " ".join(details).casefold()
    if _NOT_OVERFLOW.search(d):
        return False
    return st in (400, 413) and any(m in d for m in CONTEXT_OVERFLOW_MARKERS)


def is_silent_overflow(usage: dict[str, Any] | None, finish: str | None, window: int, text: str, tool_calls: list) -> bool:
    """A reply the provider returned as a success although the prompt did not fit (PA ``overflow.rs:73-98``): the input
    exceeds the window (the server truncated it), or ``finish == "length"`` with no output and input ≥ 0.99·W (MiMo-style
    servers). With no completion count in the usage, "no output" means no text and no tool calls. The reply is then
    discarded and the context compacted (R3)."""
    got = _usage_tokens(usage)
    if got is None or window <= 0:
        return False
    inp, out = got
    if inp > window:
        return True
    if finish == "length":
        u = usage or {}
        has_count = isinstance(u.get("completion_tokens", u.get("output_tokens")), (int, float))
        empty = out == 0 if has_count else (not (text or "").strip() and not tool_calls)
        return empty and inp >= 0.99 * window
    return False


# ---- the summarizer's view --------------------------------------------------------------------------------------------
REASONING_KEYS = ("reasoning_content", "reasoning", "reasoning_details", "thinking", "thinking_blocks")
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
KEEP_IMAGES = 3  # NG / Harbor keep the newest 3 images before an LLM summary (R16)


def _clip(text: str, head: int, tail: int, what: str = "characters") -> str:
    if len(text) <= head + tail + 80:
        return text
    cut = len(text) - head - tail
    return f"{text[:head]}\n[... {cut} {what} omitted here; the ledger holds the full text ...]\n{text[len(text) - tail:] if tail else ''}"


def _clip_args(raw: Any, head: int, tail: int) -> Any:
    """Tool-call arguments with long string values clipped, kept valid JSON (chat templates may parse them)."""
    try:
        val = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return _clip(raw, head, tail) if isinstance(raw, str) else raw
    if not isinstance(val, dict):
        return raw
    out = {k: _clip(v, head, tail) if isinstance(v, str) else v for k, v in val.items()}
    return json.dumps(out, ensure_ascii=False) if isinstance(raw, str) else out


def _pair_sanitize(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop tool rows with no preceding call and tool calls with no result (providers refuse either)."""
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    out: list[dict[str, Any]] = []
    called: set[Any] = set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            keep = [tc for tc in m["tool_calls"] if tc.get("id") in answered]
            m = {**m, "tool_calls": keep} if keep else {k: v for k, v in m.items() if k != "tool_calls"}
            called.update(tc.get("id") for tc in keep)
        elif m.get("role") == "tool" and m.get("tool_call_id") not in called:
            continue
        out.append(m)
    return out


def summary_view(messages: list[dict[str, Any]], spec: ContextSpec) -> list[dict[str, Any]]:
    """What the summarizer reads (R4, R16): a copy of ``messages`` with every tool output cut to ``tool_chars`` head +
    ``tool_tail`` tail (with a marker naming the ledger), long tool-call argument strings and very long user rows clipped
    the same way (user rows at 16x, so the task and supervisor messages normally pass whole), all but the newest
    ``KEEP_IMAGES`` images replaced by a placeholder, and reasoning (provider fields and ``<think>`` blocks) dropped from
    every assistant row but the newest. Unpaired tool rows and tool calls are dropped. The lossless copy is the
    ledger's; this view exists only for the summary request (PA's summarizer saw no tool output at all,
    ``compact_session.rs:63-68``)."""
    head, tail = spec.tool_chars, spec.tool_tail
    n_images = sum(1 for m in messages if isinstance(m.get("content"), list) for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url")
    drop_images = max(0, n_images - KEEP_IMAGES)
    last_assistant = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    out: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        role = m.get("role")
        m = dict(m)
        c = m.get("content")
        if role == "tool" and isinstance(c, str):
            m["content"] = _clip(c, head, tail)
        elif role == "user" and isinstance(c, str):
            m["content"] = _clip(c, 16 * head, 4 * tail)
        if isinstance(c, list):
            parts = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url" and drop_images > 0:
                    drop_images -= 1
                    parts.append({"type": "text", "text": "[image omitted from the summary view; the ledger keeps it]"})
                elif isinstance(p, dict) and p.get("type") == "text" and role in ("tool", "user"):
                    k = 1 if role == "tool" else 16
                    parts.append({**p, "text": _clip(str(p.get("text") or ""), k * head, (1 if role == "tool" else 4) * tail)})
                else:
                    parts.append(p)
            m["content"] = parts
        if role == "assistant":
            if m.get("tool_calls"):
                m["tool_calls"] = [{**tc, "function": {**tc["function"], "arguments": _clip_args(tc["function"].get("arguments"), head, tail)}}
                                   if isinstance(tc.get("function"), dict) else tc for tc in m["tool_calls"]]
            if i != last_assistant:
                for k in REASONING_KEYS:
                    m.pop(k, None)
                if isinstance(m.get("content"), str):
                    m["content"] = _THINK.sub("", m["content"])
        out.append(m)
    return _pair_sanitize(out)


# ---- hollowing and the tail -------------------------------------------------------------------------------------------
_HOLLOW = re.compile(r"^\[hollowed: (\d+) earlier messages omitted")


def _hollow_marker(n: int) -> dict[str, Any]:
    return {"role": "user", "content": (f"[hollowed: {n} earlier messages omitted here to fit the summary request; the ledger holds every one "
                                        "of them] Summarize only the information provided. Do not call tools.")}


def _groups(messages: list[dict[str, Any]]) -> tuple[list[list[dict[str, Any]]], int]:
    """Whole exchanges: a row plus, after an assistant row with tool calls, every row up to the last of its results (a
    message that arrives between a call and its result stays inside the exchange, so a result is never orphaned).
    Earlier hollow markers are dropped; their counts are returned."""
    groups: list[list[dict[str, Any]]] = []
    open_ids: set[Any] = set()
    prior = 0
    for m in messages:
        c = m.get("content")
        hit = _HOLLOW.match(c) if m.get("role") == "user" and isinstance(c, str) else None
        if hit:
            prior += int(hit.group(1))
            continue
        if m.get("role") == "tool" and groups:
            groups[-1].append(m)
            open_ids.discard(m.get("tool_call_id"))
            continue
        if open_ids and groups:
            groups[-1].append(m)
            if m.get("role") == "assistant" and m.get("tool_calls"):
                open_ids |= {tc.get("id") for tc in m["tool_calls"]}
            continue
        groups.append([m])
        if m.get("role") == "assistant" and m.get("tool_calls"):
            open_ids = {tc.get("id") for tc in m["tool_calls"]}
    return groups, prior


def hollow_middle(messages: list[dict[str, Any]], target_tokens: int, over: float = 1.25, *, measured: int | None = None) -> list[dict[str, Any]]:
    """Shrink the summary request's input to about ``target_tokens`` (port of nano-rlm ``compaction.py:175-223``, MIT):
    keep the leading system rows, the rows before the first assistant turn (task, summary, state) and the last exchange;
    remove whole exchanges from the middle outward, taking from the larger side, ``over`` x the excess (nano-rlm's 1.25:
    another overflowing attempt costs more than a slightly smaller input); one marker row says how many messages were
    removed and that the ledger holds them. ``measured``: the provider's token count for ``messages`` when known (calibrates
    the chars/4 sizes; ``target_tokens`` is then in the provider's units). Returns ``messages`` unchanged when it already
    fits or nothing removable is left."""
    total = approx_tokens(messages)
    scale = (measured / total) if measured and total else 1.0
    if total * scale <= target_tokens:
        return messages
    k = 0
    while k < len(messages) and messages[k].get("role") == "system":
        k += 1
    system = messages[:k]
    groups, prior = _groups(messages[k:])
    first_asst = next((i for i, g in enumerate(groups) if g[0].get("role") == "assistant"), len(groups))
    lo, hi = max(1, first_asst), len(groups) - 2  # removable group range
    if hi < lo:
        return messages
    sizes = [approx_tokens(g) for g in groups]
    need = over * (total * scale - target_tokens) / scale + approx_tokens([_hollow_marker(10**6)])
    left = right = (lo + hi) // 2
    removed = sizes[left]
    while removed < need and (left > lo or right < hi):
        if left > lo and (right == hi or sum(sizes[:left]) > sum(sizes[right + 1:])):
            left -= 1
            removed += sizes[left]
        else:
            right += 1
            removed += sizes[right]
    n = prior + sum(len(groups[j]) for j in range(left, right + 1))
    return [*system, *(m for g in groups[:left] for m in g), _hollow_marker(n), *(m for g in groups[right + 1:] for m in g)]


def split_tail(messages: list[dict[str, Any]], tail_tokens: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(head, tail)`` with the tail the longest suffix of at most ``tail_tokens`` (estimate) that starts at an assistant
    row and leaves no tool result apart from its call (PA ``compaction.rs:297-432``: never cut at a tool result). An empty
    tail when ``tail_tokens`` is 0 or no such suffix fits."""
    if tail_tokens <= 0 or not messages:
        return list(messages), []
    n = len(messages)
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + approx_tokens([messages[i]])
    owner: dict[Any, int] = {}  # tool_call_id -> index of the assistant row that called it
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                owner.setdefault(tc.get("id"), i)
    # min_owner[i]: the earliest calling row among the tool rows at index >= i (a cut at i must not pass it); an orphan
    # tool row (no caller) pins the cut after itself
    earliest = n
    min_owner = [n] * (n + 1)
    for i in range(n - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "tool":
            earliest = min(earliest, owner.get(m.get("tool_call_id"), i))
        min_owner[i] = earliest
    for i in range(n):  # the earliest valid cut that fits is the longest tail
        if messages[i].get("role") == "assistant" and min_owner[i] >= i and suffix[i] <= tail_tokens:
            return list(messages[:i]), list(messages[i:])
    return list(messages), []


# ---- the summary prompt -----------------------------------------------------------------------------------------------
_SECTIONS = """1. Goal and done-criteria: what the task asks for and how you will know it is complete.
2. Constraints and rules in force: every rule, policy, limit or preference that changes how you act, each with when it was set or changed (the step or the message) and what it replaced.
3. User and supervisor messages: every message from a person, the task giver or a supervisor, verbatim, with its sender and when it arrived.
4. Entities and ids touched: the exact ids, names, amounts and values you used or still need, each with its last observed state.
5. Pending and waiting items: what is started but not finished, and what you are waiting for, each with the trigger that should make you act.
6. Errors and fixes: what failed, why, and what fixed it (or that it is still unresolved).
7. Current hypothesis: what you currently believe about the problem and why.
8. Next step: the next thing you were about to do, quoted verbatim from your most recent turns.
9. Variables worth remembering: the kernel variables (and functions) you will reuse, each with what it holds."""

_KERNEL_NOTE = """Your Python kernel keeps running after this summary: every variable, function and import stays available, but the cells that defined them will not be in your context, so name the ones worth reusing. A variable lost in a kernel restart (a recovery notice says so) must not be listed as available. The ledger keeps everything: every earlier message and the full output of every cell stay readable with `await history.search("...")`, `await history.messages(window=w)`, `await history.user_messages()` and `await OUT(i)` for the cell output whose context copy says `[full output: OUT(i)]`. So write the summary as an index you can act on, not a transcript."""

_RULES = """Use only what is in the conversation: do not invent results, ids or state. Mark statuses as last observed, since the World may have moved on. Keep exact ids, values and wording. Reply with the summary only, as plain text, with no tool call."""

SUMMARY_PROMPT = f"""Your context is nearly full. Write a checkpoint of your own work so far that lets you continue this task from it, using these sections:

{_SECTIONS}

{_KERNEL_NOTE}

{_RULES}"""

UPDATE_PROMPT = f"""Your context is nearly full again. Your previous checkpoint is below. Update it with the work since then: keep everything in it that still holds, add what is new, move finished items out of the pending list, record every rule that changed (with when), and replace the next step. Use the same sections:

<previous-summary>
{{previous_summary}}
</previous-summary>

{_SECTIONS}

{_KERNEL_NOTE}

{_RULES}"""


def summary_request(previous_summary: str | None = None, *, instructions: str | None = None, max_tokens: int | None = None, note: str | None = None) -> str:
    """The summary request's text: ``SUMMARY_PROMPT``, or ``UPDATE_PROMPT`` with the previous summary (PA's update mode),
    plus the policy's own instructions from ``rlm.compact(instructions)`` (PA ``compaction.rs:463-479``), a length bound,
    and a retry note."""
    if previous_summary:
        text = UPDATE_PROMPT.replace("{previous_summary}", previous_summary.strip())
    else:
        text = SUMMARY_PROMPT
    if instructions and instructions.strip():
        text += f"\n\nYour own instructions for this checkpoint (follow them while keeping the sections):\n{instructions.strip()}"
    if max_tokens:
        text += f"\n\nKeep the summary under about {max(100, int(max_tokens * 0.6))} words."
    if note:
        text += f"\n\n{note}"
    return text


# ---- the rebuilt context ----------------------------------------------------------------------------------------------
SUMMARY_PREFIX = "[compaction-summary]"
STATE_PREFIX = "[python-state]"
BOUNDARY_PREFIX = "[compaction"
HARNESS_NAMES = frozenset({"world", "rlm", "host", "history", "OUT", "CONTEXT"})


def _size(e: dict[str, Any]) -> str:
    for k, unit in (("size", ""), ("len", " items"), ("chars", " chars"), ("bytes", " bytes")):
        v = e.get(k)
        if v is not None and v != "":
            return f"{v:,}{unit}" if isinstance(v, int) and not isinstance(v, bool) else f"{v}{unit}"
    return ""


def python_state_row(entries: list[dict[str, Any]], *, lost: list[str] | tuple[str, ...] = (), max_names: int = 60, max_chars: int = 2_000) -> str:
    """The ``[python-state]`` row after a compaction (R4; PA ``compaction_arms.rs:217-226``): live names with type and size,
    capped at ``max_names`` names and ``max_chars`` characters (B10: the old ``Live names:`` line listed every global, so
    per-ticket variables put thousands of names into the prompt), then the names lost in a kernel restart. ``entries``:
    dicts with ``name``, ``type`` and one of ``size`` / ``len`` / ``chars`` / ``bytes`` (a bare string is a name), most
    relevant first; underscore names and the harness objects are left out."""
    rows: list[str] = []
    for e in entries:
        e = {"name": e} if isinstance(e, str) else e
        name = str(e.get("name") or "")
        if not name or name.startswith("_") or name in HARNESS_NAMES:
            continue
        typ, size = str(e.get("type") or ""), _size(e)
        rows.append(name + (f": {typ}" if typ else "") + (f" ({size})" if size else ""))
    head = f"{STATE_PREFIX} Your kernel kept running; these variables are live (name: type (size)):"
    lost_txt = ""
    if lost:
        shown = list(lost)[:20]
        lost_txt = (" Lost in a kernel restart (redefine before use): " + ", ".join(shown)
                    + (f" and {len(lost) - len(shown)} more" if len(lost) > len(shown) else "") + ".")
        lost_txt = lost_txt[: max_chars // 3]
    more_tpl = " ... and {k} more (list them with `[n for n in globals() if not n.startswith('_')]`)."
    budget = max_chars - len(head) - len(lost_txt) - len(more_tpl) - 8
    shown_rows: list[str] = []
    used = 0
    for r in rows[:max_names]:
        if used + len(r) + 2 > budget:
            break
        shown_rows.append(r)
        used += len(r) + 2
    body = " " + ", ".join(shown_rows) if shown_rows else " (none)"
    rest = len(rows) - len(shown_rows)
    text = head + body + (more_tpl.format(k=rest) if rest else ".") + lost_txt
    return text[:max_chars]


def boundary_marker(n: int, before: int, after: int, window: int) -> str:
    """The boundary row (MiMo ``surgery.py:108-120``; H3): the compaction count, the context size before and after, the
    ledger window this context opens, the history API, and the gap rule (MiMo ``surgery.py:129-130``)."""
    prev = max(0, window - 1)
    return (f"{BOUNDARY_PREFIX} {n}] Your context was compacted from about {before:,} to {after:,} tokens; this is context window {window} "
            f"of your ledger. Earlier messages and the full output of every cell are still readable: `await history.windows()`, "
            f"`await history.messages(window={prev})`, `await history.search(\"...\")`, `await history.user_messages()`, `await OUT(i)`. "
            "If you need a detail that is not in this context, state the gap before relying on missing details, then look it up in the "
            "history instead of guessing. Refresh INBOX and the state of pending items before repeating any action.")


def rebuild(system: dict[str, Any], task_msg: dict[str, Any], summary: str, state_row: str, boundary: str, tail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The context after a compaction (R4): system, the task row verbatim (never summarized), the summary framed as
    lagging behind the retained messages (PA ``messages.rs:12-17``), the ``[python-state]`` row, the boundary marker, the
    recent tail. No assistant turn is written by the harness: the only assistant rows are the tail's own, and tool rows
    whose call is not in the tail are dropped (a tail from ``split_tail`` has none)."""
    framed = (f"{SUMMARY_PREFIX}\nThe conversation before this point was compacted into the summary below, which you wrote. The messages "
              "retained after it are authoritative; this summary may lag behind them.\n\n<summary>\n"
              f"{summary.strip()}\n</summary>\n\nCompaction did not finish, cancel or restart anything: treat the statuses above as last "
              "observed. Do not redo work merely because its full conversation is absent from this context.")
    rows: list[dict[str, Any]] = [dict(system), dict(task_msg), {"role": "user", "content": framed}]
    if state_row:
        rows.append({"role": "user", "content": state_row})
    if boundary:
        rows.append({"role": "user", "content": boundary})
    called: set[Any] = set()
    for m in tail:
        if m.get("role") == "assistant":
            called.update(tc.get("id") for tc in m.get("tool_calls") or [])
        elif m.get("role") == "tool" and m.get("tool_call_id") not in called:
            continue
        rows.append(m)
    return rows


# ---- the summary loop -------------------------------------------------------------------------------------------------
@dataclass
class CompactionOutcome:
    ok: bool
    summary: str
    attempts: int
    failure: str | None        # None | "compaction_failed"
    input_tokens: int          # the last attempt's input (provider usage when reported, else the calibrated estimate)
    detail: str = ""           # why the last rejected attempt was rejected (empty / tool_call / length / overflow / ...)


def _clean_summary(text: str) -> str:
    """The reply's final text: ``<think>`` blocks removed (reasoning never enters a summary; nano-rlm ``engine.py:1318-1321``)."""
    t = _THINK.sub("", text or "")
    if "<think>" in t:  # an unterminated thinking block: the reply is reasoning only
        t = t.split("<think>", 1)[0]
    return t.strip()


OnCall = Callable[[Any, int, "bool | None", int], None]


class Compactor:
    """The summary loop (R3; port of nano-rlm ``engine.py:1250-1365``, MIT). ``run`` asks the model for a checkpoint of the
    ``summary_view`` of ``messages``, hollowed to ``spec.summary_input_tokens()``:

    - at most ``spec.max_compaction_attempts`` requests;
    - an overflow (raised, or silent per ``is_silent_overflow``) → ``hollow_middle`` the input (≥ ``reserve`` or 20 %
      less, ×1.25 over-removal) and retry; when it cannot shrink further the loop stops;
    - an empty, tool-calling, reasoning-only or ``length``-truncated reply is rejected and resampled with a note (the
      note changes the request, so a temperature-0 model does not repeat itself; the seed changes too, so a record /
      replay cache cannot return the rejected reply); a truncated reply whose input was over budget is hollowed instead;
    - a provider fault (``LLMError`` / ``BudgetExceeded`` / anything else that is not an overflow) propagates after it is
      reported: the loop masks it with its own ``failure_kind``;
    - every attempt is reported through ``on_call(response_or_exception, attempt, accepted, input_tokens)``: ``accepted``
      True (the summary the run continues from: the only trainable attempt, verifiers ``trace.py:515-553``), False (a
      reply that was rejected), None (no reply: an exception);
    - all attempts fail → ``CompactionOutcome(ok=False, failure="compaction_failed")``: the loop's clean typed stop
      (truncation, not error; nano-rlm ``engine.py:762-768``).

    ``meter``: the agent's ``ContextMeter``; its calibration ratio converts token budgets into chars/4 units."""

    def __init__(self, *, meter: ContextMeter | None = None):
        self.meter = meter

    def _ratio(self) -> float:
        return max(1.0, self.meter.ratio) if self.meter is not None else 1.0

    async def run(self, llm: Any, messages: list[dict[str, Any]], spec: ContextSpec, *, previous_summary: str | None = None,
                  on_call: OnCall | None = None, sample_key: str | None = None, temperature: float = 0.0,
                  instructions: str | None = None) -> CompactionOutcome:
        report = on_call or (lambda *a: None)
        src = messages
        if previous_summary:  # update mode: the old summary row is in <previous-summary>, not twice
            src = [m for m in messages if not (m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].startswith(SUMMARY_PREFIX))]
        view = summary_view(src, spec)
        ratio = self._ratio()
        out_tokens = spec.summary_tokens()
        budget = spec.summary_input_tokens()
        note: str | None = None

        def prompt_msg() -> dict[str, Any]:
            return {"role": "user", "content": summary_request(previous_summary, instructions=instructions, max_tokens=out_tokens, note=note)}

        # the request must fit summary_input_tokens(): view (calibrated estimate) + the prompt (B1 / threshold() docstring)
        view = hollow_middle(view, max(1, int((budget - approx_tokens([prompt_msg()]) * ratio) / ratio)))
        last_in, detail = 0, ""
        attempt = 0
        for attempt in range(1, spec.max_compaction_attempts + 1):
            req = [*view, prompt_msg()]
            est_in = int(approx_tokens(req) * ratio)
            seed = (0 if temperature == 0 else None) if attempt == 1 else attempt - 1
            try:
                r = await llm.complete(req, tools=None, max_tokens=out_tokens, temperature=temperature, sample_key=sample_key, seed=seed)
            except Exception as e:
                report(e, attempt, None, est_in)
                if not is_overflow(e):
                    raise  # LLMError / BudgetExceeded / a bug: the loop masks it with its kind
                last_in, detail = est_in, "overflow"
                smaller = self._shrink(view, est_in, spec, ratio)
                if smaller is view:
                    break
                view = smaller
                continue
            got = _usage_tokens(r.usage)
            in_tok = got[0] if got else est_in
            last_in = in_tok
            if is_silent_overflow(r.usage, r.finish, spec.window_tokens, r.text, r.tool_calls):
                report(r, attempt, False, in_tok)
                detail = "silent_overflow"
                smaller = self._shrink(view, in_tok, spec, ratio)
                if smaller is view:
                    break
                view = smaller
                continue
            text = _clean_summary(r.text)
            if r.tool_calls:
                detail, note = "tool_call", "Your previous reply called a tool. Reply with the summary as plain text only; no tool is available."
            elif not text:
                detail, note = "empty", "Your previous reply had no summary text. Reply with the summary as plain text."
            elif r.finish == "length":
                detail, note = "length", "Your previous summary was cut off at the output limit. Write a shorter one: keep the sections, drop detail the ledger already holds."
            else:
                report(r, attempt, True, in_tok)
                return CompactionOutcome(ok=True, summary=text, attempts=attempt, failure=None, input_tokens=in_tok)
            report(r, attempt, False, in_tok)
            if detail == "length" and got and got[0] > budget:  # cut off because the input crowded the window: hollow, measured
                per = got[0] / max(1, approx_tokens(req))  # provider tokens per chars/4 token on this very request
                view_real = int(approx_tokens(view) * per)
                view = hollow_middle(view, max(1, budget - int(approx_tokens([prompt_msg()]) * per)), measured=view_real)
        return CompactionOutcome(ok=False, summary="", attempts=attempt, failure="compaction_failed", input_tokens=last_in, detail=detail)

    @staticmethod
    def _shrink(view: list[dict[str, Any]], in_tokens: int, spec: ContextSpec, ratio: float) -> list[dict[str, Any]]:
        """After an overflow: remove at least ``reserve`` tokens or 20 % of the input, whichever is more (nano-rlm removes
        ``RESERVE_TOKENS`` per overflow, ``engine.py:1304-1311``; 20 % keeps a 1M window from needing dozens of tries).
        Returns ``view`` itself when nothing more can go."""
        est = approx_tokens(view)
        cur = int(est * ratio)
        target = min(cur - spec.reserve, int(cur * 0.8), int(in_tokens * 0.8))
        smaller = hollow_middle(view, max(1, int(target / ratio)))
        return view if approx_tokens(smaller) >= est else smaller


# ---- what the model sees about its budgets ----------------------------------------------------------------------------
def footer(*, used: int, threshold: int, window: int, step: int, turns: int, budgets: dict[str, tuple[int, int]], elapsed_s: float | None,
           source: str, compact_hint: bool = False) -> str:
    """The ``<context_usage>`` footer on every observation from ``spec.footer_from`` of the threshold (MiMo
    ``cc_agent.py:227-244``; H15): World steps and turns, context used against the compaction threshold and the window
    (with its source), each budget as ``left of limit`` (``budgets[name] = (left, limit)``), and the last cell's elapsed
    time. Factual, no realism pretence; ``compact_hint`` (mode ``action``) adds that ``rlm.compact()`` frees context."""
    pct = round(100 * used / threshold) if threshold > 0 else 0
    parts = [f"step {step:,}, turn {turns:,}",
             f"context {used:,} of {threshold:,} tokens before compaction ({pct}%; window {window:,}; source: {source})"]
    if budgets:
        parts.append("left: " + ", ".join(f"{k} {max(0, int(v[0])):,} of {int(v[1]):,}" for k, v in budgets.items()))
    if elapsed_s is not None:
        parts.append(f"last cell {elapsed_s:.1f} s")
    text = "<context_usage>" + "; ".join(parts) + "."
    if compact_hint:
        text += " `rlm.compact()` writes a checkpoint and frees context; compact at a natural boundary when substantial work remains."
    return text + "</context_usage>"


_UNITS = {"turns": "turns", "tokens": "tokens", "new_tokens": "tokens", "wall_s": "seconds", "time": "seconds", "steps": "World steps",
          "sub_calls": "sub-calls", "spawns": "child agents", "cost_usd": "USD"}


def wrap_up_notice(kind: str, left: int) -> str:
    """At 0.9 of an episode budget (H3): finish now, do not compact (a compaction spends what is left on housekeeping)."""
    unit = _UNITS.get(kind, kind)
    return (f"[budget: {kind}] {left:,} {unit} left of this episode's {kind} budget. Finish now: complete the essential remaining work, check "
            "it against every requirement in TASK, and call rlm.finish(summary). Do not start new exploration and do not compact.")
