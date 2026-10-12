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
- ``HFVisionChatCodec``: a vision-language processor (Qwen2-VL / Qwen2.5-VL style): images in the messages are expanded
  into placeholder runs by the processor itself, and every render / suffix reports the images it expanded
  (``last_media``: content-addressed refs, placeholder offset and length, the patch grid). Pixels never enter the
  record; ``train.vision.VisionHFPolicy`` rebuilds them from the refs for log-prob recomputation.
"""

from __future__ import annotations

import base64
import hashlib
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

    def thinking_behaviour(self) -> str:
        return "preserved"  # renders every message verbatim


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

    def thinking_behaviour(self) -> str:
        """What this chat template does with earlier reasoning (m13 C-7, A2-48), probed by rendering: "preserved" (kept
        everywhere), "interleaved" (kept for assistant turns after the last user message, i.e. inside the current tool
        loop, dropped once a new user turn arrives: Qwen3, DeepSeek V3.x), "strip" (always dropped)."""
        mark = "zq7reasoningmark"
        a = {"role": "assistant", "content": f"<think>{mark}</think>answer"}

        def has(msgs: list[dict[str, Any]]) -> bool:
            return mark in self.tok.decode(self._apply(msgs, None, gen=True))

        before_user = has([{"role": "user", "content": "q"}, a, {"role": "user", "content": "q2"}])
        try:
            in_loop = has([{"role": "user", "content": "q"}, {**a, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                           {"role": "tool", "tool_call_id": "c1", "content": "r"}])
        except Exception:  # noqa: BLE001 -- a template without tool turns: reasoning after the last user message, plainly
            in_loop = has([{"role": "user", "content": "q"}, a])
        return "preserved" if before_user else ("interleaved" if in_loop else "strip")


# ---- vision-language codec ---------------------------------------------------------------------------------------------
MEDIA_PREFIX = "sha256:"  # the World store's content address (``core.ids.HASH_PREFIX``): a record's image ref is the same key


class MemoryMediaStore:
    """The default image store of ``HFVisionChatCodec``: content-addressed bytes in memory (``put_bytes`` / ``get_bytes``,
    the ``core.store.Store`` surface, which serves as a durable drop-in)."""

    def __init__(self) -> None:
        self._b: dict[str, bytes] = {}

    def put_bytes(self, data: bytes) -> str:
        key = MEDIA_PREFIX + hashlib.sha256(data).hexdigest()
        self._b[key] = data
        return key

    def get_bytes(self, key: str) -> bytes:
        return self._b[key]


def decode_data_url(url: str) -> tuple[str, bytes]:
    """``data:<mime>;base64,<payload>`` -> (mime, bytes). Remote URLs are refused: an image must be content-addressed (in
    the store) before it can be trained on, or the trainer could see different pixels than the engine did."""
    header, sep, payload = url.partition(",")
    if not sep or not header.startswith("data:") or ";base64" not in header:
        raise ValueError("vision training needs base64 data URLs or stored refs, not remote image URLs (fetch them into the store first)")
    return header[5:].split(";", 1)[0] or "image/png", base64.b64decode(payload, validate=True)


def load_image(data: bytes) -> Any:
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(data)) as im:
        return im.convert("RGB")


def placeholder_runs(ids: list[int], token_id: int) -> list[tuple[int, int]]:
    """(offset, length) of every maximal run of ``token_id`` (one run per image: runs are separated by vision start / end
    tokens)."""
    out: list[tuple[int, int]] = []
    i, n = 0, len(ids)
    while i < n:
        if ids[i] == token_id:
            j = i
            while j < n and ids[j] == token_id:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


class HFVisionChatCodec:
    """A Hugging Face vision-language processor (``processor.tokenizer``, ``processor.image_processor``, the processor's
    own chat template) with exact TITO.

    - ``render``: the processor's chat template to text (image blocks become the template's image pad), then the processor
      itself on (text, images): it expands each pad into ``prod(grid) / merge_size**2`` placeholder tokens, exactly as the
      engine's processor does (vLLM runs the same HF processor), so the ids are the engine's ids.
    - ``suffix``: the sentinel technique of ``HFChatCodec`` on the processor output: render [dummy user, sentinel
      assistant] + new messages with only the new messages' images and cut after the end-of-turn token closing the
      sentinel. The previous prompt and completion are never re-tokenized, so behaviour log-probs stay valid across a turn
      that brings an image (a tool screenshot, an attachment).
    - ``last_media``: the images the last render / suffix expanded, in order: ``{kind, mime, ref, offset, length, grid}``
      with ``offset`` relative to the ids that call returned (``EngineLLM`` makes it absolute). ``ref`` is the sha256 of
      the image bytes, the World store's key for the same image (``World.put_media``), and the bytes are kept in
      ``store`` so the trainer can rebuild the pixels. Every placeholder run is checked against the processor's grid
      (prime-rl's materialize check), so a template / processor disagreement fails loudly here, not in the trainer.

    Image blocks: OpenAI ``image_url`` parts with a data URL (what ``LLMToolPolicy(vision=True)`` sends for
    ``Observation.instruction_media``, message attachments and ``ToolResult.images``), or ``{"type": "image", "ref":
    "sha256:..."}`` for an image already in the store. ``processor_kwargs`` (min / max pixels) are fixed for the codec's
    life: a renderer whose resizing changed between turns would change the expansion.

    Pattern: verl ``utils/tokenizer/continuous_token.py`` (``VLContinuousTokenMixin``: processor-backed render, media bound
    per message, dummy + trim for incremental turns), prime-rl ``multimodal/qwen_vl.py`` (placeholder length = T·H·W /
    merge², checked against the engine's placeholder ranges) and ``trainer/multimodal.py`` (images materialized at
    train time from references). Written anew; refs are content addresses instead of data URLs.
    """

    SENTINEL = HFChatCodec.SENTINEL

    def __init__(self, processor: Any, *, template_kwargs: dict | None = None, processor_kwargs: dict | None = None, store: Any = None,
                 image_token: str | None = None):
        self.proc, self.tok, self.image_processor = processor, processor.tokenizer, processor.image_processor
        self.kw, self.pkw = dict(template_kwargs or {}), dict(processor_kwargs or {})
        eos = self.tok.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        tid = getattr(processor, "image_token_id", None)
        name = image_token or getattr(processor, "image_token", None) or "<|image_pad|>"
        self.image_token_id = int(tid if tid is not None and image_token is None else self.tok.convert_tokens_to_ids(name))
        self.merge = int(getattr(self.image_processor, "merge_size", 1) or 1)
        self.store = store if store is not None else MemoryMediaStore()
        self.last_media: list[dict[str, Any]] = []

    # ---- media -----------------------------------------------------------------------------------------------------
    def load(self, ref: str) -> bytes:
        return self.store.get_bytes(ref)

    def _norm(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
        """Messages for the template (image blocks reduced to ``{"type": "image"}``) and the images in order: (mime, ref)."""
        out, imgs = [], []
        for m in messages:
            c = m.get("content")
            if not isinstance(c, list):
                out.append(m)
                continue
            parts = []
            for p in c:
                kind = p.get("type") if isinstance(p, dict) else None
                if kind == "image_url":
                    url = p["image_url"]["url"] if isinstance(p.get("image_url"), dict) else p.get("image_url")
                    mime, data = decode_data_url(str(url))
                    imgs.append((mime, self.store.put_bytes(data)))
                    parts.append({"type": "image"})
                elif kind == "image" and p.get("ref"):
                    self.store.get_bytes(p["ref"])  # must already be stored: fail here, not in the trainer
                    imgs.append((p.get("mime", "image/png"), p["ref"]))
                    parts.append({"type": "image"})
                elif kind == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
                else:
                    raise ValueError(f"HFVisionChatCodec handles text and image parts, not {kind!r}")
            out.append({**m, "content": parts})
        return out, imgs

    def _apply(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, *, gen: bool) -> tuple[list[int], list[dict[str, Any]]]:
        norm, imgs = self._norm(messages)
        text = self.proc.apply_chat_template(norm, tools=tools, add_generation_prompt=gen, tokenize=False, **self.kw)
        if isinstance(text, list):
            text = text[0]
        pil = [load_image(self.store.get_bytes(ref)) for _, ref in imgs]
        enc = self.proc(text=[text], images=pil or None, return_tensors="pt", **self.pkw)
        ids = enc["input_ids"]
        ids = [int(x) for x in (ids[0].tolist() if hasattr(ids, "tolist") else ids[0])]
        runs = placeholder_runs(ids, self.image_token_id)
        if len(runs) != len(imgs):
            raise ValueError(f"the processor expanded {len(runs)} image placeholder runs for {len(imgs)} images")
        grids = enc["image_grid_thw"].tolist() if imgs else []
        media = []
        for (off, n), (mime, ref), g in zip(runs, imgs, grids, strict=True):
            want = (int(g[0]) * int(g[1]) * int(g[2])) // (self.merge * self.merge)
            if n != want:
                raise ValueError(f"image {ref[:19]}: {n} placeholder tokens, the grid {g} needs {want}")
            media.append({"kind": "image", "mime": mime, "ref": ref, "offset": off, "length": n, "grid": [int(x) for x in g]})
        return ids, media

    # ---- the codec protocol ----------------------------------------------------------------------------------------
    def render(self, messages, tools) -> list[int]:
        ids, self.last_media = self._apply(messages, tools, gen=True)
        return ids

    def suffix(self, new_messages, tools) -> list[int]:
        dummy = [{"role": "user", "content": "x"}, {"role": "assistant", "content": self.SENTINEL}]
        full, media = self._apply(dummy + list(new_messages), None, gen=True)
        sent = self.tok.encode(self.SENTINEL, add_special_tokens=False)
        start = next((i for i in range(len(full) - len(sent) + 1) if full[i:i + len(sent)] == sent), None)
        if start is None:
            raise ValueError("chat template dropped the sentinel assistant turn; cannot compute the exact-TITO suffix")
        end = next((i for i in range(start + len(sent), len(full)) if full[i] in self.eos_ids), None)
        if end is None:
            raise ValueError("chat template closes assistant turns without an EOS token; cannot compute the exact-TITO suffix")
        cut = end + 1
        self.last_media = [{**m, "offset": m["offset"] - cut} for m in media]  # the dummy has no images: all of them follow the cut
        return full[cut:]

    def parse(self, completion_ids: list[int]) -> tuple[str, list[dict[str, Any]]]:
        return parse_tool_calls(self.tok.decode(completion_ids, skip_special_tokens=True))

    def thinking_behaviour(self) -> str:
        """As ``HFChatCodec.thinking_behaviour``, probed on the processor's template."""
        mark = "zq7reasoningmark"
        a = {"role": "assistant", "content": f"<think>{mark}</think>answer"}

        def has(msgs: list[dict[str, Any]]) -> bool:
            text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **self.kw)
            return mark in (text[0] if isinstance(text, list) else text)

        before_user = has([{"role": "user", "content": "q"}, a, {"role": "user", "content": "q2"}])
        try:
            in_loop = has([{"role": "user", "content": "q"}, {**a, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                           {"role": "tool", "tool_call_id": "c1", "content": "r"}])
        except Exception:  # noqa: BLE001 -- a template without tool turns
            in_loop = has([{"role": "user", "content": "q"}, a])
        return "preserved" if before_user else ("interleaved" if in_loop else "strip")
