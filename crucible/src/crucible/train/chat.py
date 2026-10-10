"""Chat codecs: messages ↔ token ids for the engine, with exact token-in-token-out across turns (research m11-m12/04
§C.4 TRL ``_get_tool_suffix_ids``; m11-m12/01 slime CLEAN/REALIGN/FORK; m11-m12/02 ROLL ``MessageTracker``).

The rule that keeps behaviour logprobs valid: **the model's own output is never re-tokenized.** Turn k+1's prompt is

    prompt_k + completion_k (+ EOS if the engine stopped before emitting it) + suffix(new messages)

where ``suffix`` renders only the messages that arrived after the assistant turn (tool results, user / actor messages),
cut from a render of a fixed dummy conversation, so the template's per-message framing is exact. A context the harness
rebuilt (compaction) is rendered whole and starts a new segment.

- ``ChatCodec`` protocol: ``render(messages, tools)``, ``suffix(new_messages)``, ``parse(completion_ids)``, ``eos_ids``.
- ``HFChatCodec``: any HF tokenizer with a chat template; Hermes/Qwen ``<tool_call>{json}</tool_call>`` parsing.
- ``ByteChatCodec``: a byte-level codec with a tiny template for CPU tests (vocab 256 + specials).
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from typing import Any, Protocol

TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    calls = []
    for m in TOOL_CALL.finditer(text):
        try:
            d = json.loads(m.group(1))
            calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "name": d.get("name", ""), "arguments": d.get("arguments") or {}})
        except json.JSONDecodeError:
            calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "name": "", "arguments": {"__unparseable__": m.group(1)}})
    return TOOL_CALL.sub("", text).strip(), calls


class ChatCodec(Protocol):
    eos_ids: list[int]

    def render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[int]: ...
    def suffix(self, new_messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[int]: ...
    def parse(self, completion_ids: list[int]) -> tuple[str, list[dict[str, Any]]]: ...


class ByteChatCodec:
    """Bytes 0..255; 256 = <|eos|> (end of a message), 257 = <|bos|>. Template: <bos>role\\ncontent<eos> per message,
    tools listed in the first system message, generation prompt ``<bos>assistant\\n``."""

    EOS, BOS = 256, 257
    vocab = 258

    def __init__(self) -> None:
        self.eos_ids = [self.EOS]

    def _msg(self, m: dict[str, Any]) -> list[int]:
        content = m.get("content") or ""
        if m.get("tool_calls"):
            content += "".join(f"<tool_call>{json.dumps({'name': c['function']['name'], 'arguments': json.loads(c['function']['arguments'] or '{}')})}</tool_call>"
                               for c in m["tool_calls"])
        return [self.BOS, *f"{m['role']}\n{content}".encode(), self.EOS]

    def render(self, messages, tools) -> list[int]:
        out: list[int] = []
        if tools:
            out += [self.BOS, *("tools\n" + json.dumps([t["function"]["name"] for t in tools])).encode(), self.EOS]
        for m in messages:
            out += self._msg(m)
        return out + [self.BOS, *b"assistant\n"]

    def suffix(self, new_messages, tools) -> list[int]:
        out: list[int] = []
        for m in new_messages:
            out += self._msg(m)
        return out + [self.BOS, *b"assistant\n"]

    def encode_text(self, text: str) -> list[int]:
        return list(text.encode())

    def parse(self, completion_ids: list[int]) -> tuple[str, list[dict[str, Any]]]:
        text = bytes(t for t in completion_ids if t < 256).decode(errors="replace")
        return parse_tool_calls(text)


class HFChatCodec:
    """A Hugging Face tokenizer's chat template. ``suffix`` = render([dummy user, dummy assistant] + new, gen prompt)
    minus render([dummy user, dummy assistant]): the exact framing of the new messages (TRL's technique)."""

    def __init__(self, tokenizer: Any, *, template_kwargs: dict | None = None):
        self.tok = tokenizer
        self.kw = dict(template_kwargs or {})
        eos = tokenizer.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        self._dummy = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
        self._dummy_ids = self._apply(self._dummy, None, gen=False)

    def _apply(self, messages, tools, *, gen: bool) -> list[int]:
        ids = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=gen, tokenize=True, return_dict=False, **self.kw)
        # transformers 5 returns a BatchEncoding (a UserDict, not a dict) by default: a list of its *keys* was taken as the
        # prompt token ids and every real-tokenizer rollout failed (found by the M12 GPU gate: 228k masked rollouts, 0 steps)
        if isinstance(ids, Mapping) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        ids = list(ids)
        if ids and isinstance(ids[0], list):  # batched form
            ids = ids[0]
        if not all(isinstance(i, int) for i in ids):
            raise TypeError(f"chat template did not return token ids: {type(ids[0]).__name__ if ids else 'empty'}")
        return ids

    def render(self, messages, tools) -> list[int]:
        return self._apply(messages, tools, gen=True)

    SENTINEL = "\u27e6crucible-turn-end\u27e7"

    def suffix(self, new_messages, tools) -> list[int]:
        """The tokens the template appends *after* an assistant turn ends, for the messages that follow it: exact TITO
        (next prompt = the sampled prompt + completion + EOS + this suffix).

        Taken after the end-of-turn token that closes a sentinel assistant message, not as a diff against a rendered
        dummy: templates re-render history when a turn is appended (Qwen3 gives only the *last* assistant turn an empty
        ``<think></think>`` block and strips it from earlier ones), so no rendered dummy is a prefix of the longer render
        (found on the lab: every multi-turn Qwen3 episode raised "not prefix-preserving")."""
        dummy = [{"role": "user", "content": "x"}, {"role": "assistant", "content": self.SENTINEL}]
        full = self._apply(dummy + list(new_messages), None, gen=True)
        sent = self.tok.encode(self.SENTINEL, add_special_tokens=False)
        start = next((i for i in range(len(full) - len(sent) + 1) if full[i:i + len(sent)] == sent), None)
        if start is None:
            raise ValueError("chat template dropped the sentinel assistant turn; cannot compute the exact-TITO suffix")
        end = next((i for i in range(start + len(sent), len(full)) if full[i] in self.eos_ids), None)
        if end is None:
            raise ValueError("chat template closes assistant turns without an EOS token; cannot compute the exact-TITO suffix")
        return full[end + 1:]

    def parse(self, completion_ids: list[int]) -> tuple[str, list[dict[str, Any]]]:
        return parse_tool_calls(self.tok.decode(completion_ids, skip_special_tokens=True))
