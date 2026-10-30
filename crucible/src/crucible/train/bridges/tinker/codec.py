"""Our renderer for Tinker models: the model's own HF chat template, exact token-in/token-out, and a tool-call parser per
family (swarm-pipeline/09 §3).

``TinkerChatCodec`` is ``train.chat.HFChatCodec`` (exact TITO: turn k+1's prompt = prompt_k + completion_k (+ EOS) +
the template's framing of the new messages, never a re-render of the model's own tokens) with three changes the
Qwen3.5 family needed on its first contact (probe of 2026-10-20 on the Qwen3.5 / Qwen3.6 tokenizer, one template):

1. **Tool-call arguments as mappings.** The Qwen3.5 / 3.6 / 3.8 templates iterate ``tool_call.arguments|items``, and the
   policy stores arguments as a JSON string (the OpenAI form): every render of a history with a tool call raised
   ``TypeError: Can only get item pairs from a mapping``. ``_apply_raw`` decodes them first (a string that is not a JSON
   object is wrapped as ``{"arguments": <raw>}``).
2. **The family's tool format.** Qwen3.5 does not emit Hermes JSON: it emits
   ``<tool_call>\\n<function=name>\\n<parameter=p>\\nvalue\\n</parameter>\\n</function>\\n</tool_call>``, which
   ``HFChatCodec.parse`` (``<tool_call>{json}</tool_call>``) never matched, so no tool call would ever have run.
   ``parse`` dispatches on ``TinkerModel.tool_format``: ``qwen_xml``, ``dsml`` (DeepSeek V4.1), ``glm`` (GLM-5.3),
   ``hermes``; the parsers are ported from tinker-cookbook ``ad3b5c5`` (Apache-2.0, ``LICENSES/Apache-2.0.txt``):
   ``renderers/qwen3_5.py`` ``_FUNCTION_BLOCK_RE`` / ``_PARAM_BLOCK_RE`` and ``_parse_qwen3_5_tool_call_xml``,
   ``renderers/deepseek_v4_1.py`` ``_CALLS_BLOCK_RE`` / ``_INVOKE_RE`` / ``_PARAM_RE`` and ``_parse_tool_calls``,
   ``renderers/glm5_3.py`` ``_GLM_TOOL_CALL_RE`` / ``_GLM_ARG_PAIR_RE``. A malformed block becomes a call with
   ``{"__unparseable__": raw}`` arguments: the World answers it with an error the policy sees (M12 audit V17), never a
   silent drop. Decoding keeps special tokens (DeepSeek's ``<｜DSML｜`` markers are added tokens) and drops only the
   end-of-turn ids.
3. **Reasoning, recorded and explicit.** ``enable_thinking`` is passed to the template as the model's switch
   (``TinkerModel.thinking_kw``), default off for the student. With it on, the completion's ``<think>`` block is
   removed from the parsed text (the tokens, which are what trains, keep it). The template's history behaviour
   (``thinking_behaviour``: Qwen3.5 is "interleaved") is what ``EngineLLM`` checks against.

**The text protocol** (``tool_format="text"``, a family whose native tool format is not ported: Kimi, gpt-oss, base
models): the tools are declared in the system message as plain text with a Hermes JSON calling convention, earlier
calls render as that JSON in the assistant content and tool results as user turns; ``parse`` reads Hermes JSON.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from typing import Any

from crucible.train.bridges.tinker.models import TinkerModel
from crucible.train.chat import TOOL_CALL, HFChatCodec, _arguments

# ---- ported parsers (tinker-cookbook ad3b5c5, Apache-2.0) ------------------------------------------------------------------
_QWEN_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_BLOCK_RE = re.compile(r"^\s*<function=(?P<name>[^>\n]+)>\s*(?P<body>.*?)\s*</function>\s*$", re.DOTALL)
_PARAM_BLOCK_RE = re.compile(r"<parameter=(?P<name>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>", re.DOTALL)
_DSML_CALLS = re.compile(r"(?:\n\n)?<｜DSML｜ calls>.*?</｜DSML｜ calls>", re.DOTALL)
_DSML_INVOKE = re.compile(r'<｜DSML｜ invoke name="(.*?)">\n(.*?)</｜DSML｜ invoke>', re.DOTALL)
_DSML_PARAM = re.compile(r'<｜DSML｜ parameter name="(.*?)" string="(true|false)">(.*?)</｜DSML｜ parameter>', re.DOTALL)
_GLM_TOOL_CALL_RE = re.compile(r"^\s*(?P<name>[\w\-\.]+)\s*(?P<body>.*?)\s*$", re.DOTALL)
_GLM_ARG_PAIR_RE = re.compile(r"<arg_key>(?P<key>.*?)</arg_key>\s*<arg_value>(?P<value>.*?)</arg_value>", re.DOTALL)
_THINK = re.compile(r"^\s*(?:<think>)?(?P<reasoning>.*?)</think>\s*", re.DOTALL)


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"id": f"call_{uuid.uuid4().hex[:8]}", "name": name, "arguments": arguments}


def _bad(raw: str, why: str) -> dict[str, Any]:
    return {"id": f"call_{uuid.uuid4().hex[:8]}", "name": "", "arguments": {"__unparseable__": raw, "__error__": why}}


def _value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_qwen_xml(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Qwen3.5 / Nemotron: ``<tool_call><function=f><parameter=p>v</parameter></function></tool_call>``; a block holding
    JSON instead (a model that falls back to Hermes) is read as Hermes."""
    calls: list[dict[str, Any]] = []
    for blk in _QWEN_BLOCK.finditer(text):
        raw = blk.group(1)
        if "<function=" not in raw:
            _, hermes = _parse_hermes(blk.group(0))
            calls += hermes or [_bad(raw, "neither <function=...> nor JSON")]
            continue
        m = _FUNCTION_BLOCK_RE.match(raw)
        if not m or not m.group("name").strip():
            calls.append(_bad(raw, "malformed <function=...> block"))
            continue
        args: dict[str, Any] = {}
        pos, body, ok = 0, m.group("body"), True
        for p in _PARAM_BLOCK_RE.finditer(body):
            if body[pos:p.start()].strip() or not p.group("name").strip():
                ok = False
                break
            args[p.group("name").strip()] = _value(p.group("value").strip("\n"))
            pos = p.end()
        if not ok or body[pos:].strip():
            calls.append(_bad(raw, "text outside <parameter> blocks"))
            continue
        calls.append(_call(m.group("name").strip(), args))
    # an unterminated block (cut at max_tokens) is a call the policy tried to make: surfaced, never dropped
    tail = text[text.rfind("</tool_call>") + len("</tool_call>"):] if "</tool_call>" in text else text
    if "<tool_call>" in tail:
        calls.append(_bad(tail[tail.index("<tool_call>"):], "unterminated <tool_call> block"))
    head = text.split("<tool_call>", 1)[0]
    return head.strip(), calls


def _parse_hermes(text: str) -> tuple[str, list[dict[str, Any]]]:
    calls = []
    for m in TOOL_CALL.finditer(text):
        try:
            d = json.loads(m.group(1))
        except json.JSONDecodeError:
            calls.append(_bad(m.group(1), "tool call is not JSON"))
            continue
        name = d.get("name", "") if isinstance(d, dict) else ""
        calls.append(_call(name if isinstance(name, str) else str(name), _arguments(d.get("arguments")) if isinstance(d, dict) else {"__unparseable__": m.group(1)}))
    return TOOL_CALL.sub("", text).strip(), calls


def parse_dsml(text: str) -> tuple[str, list[dict[str, Any]]]:
    blk = _DSML_CALLS.search(text)
    if not blk:
        return text.strip(), []
    body = blk.group(0).removeprefix("\n\n").removeprefix("<｜DSML｜ calls>").removesuffix("</｜DSML｜ calls>")
    invokes = list(_DSML_INVOKE.finditer(body))
    if not invokes or _DSML_INVOKE.sub("", body).strip():
        return text[: blk.start()].strip(), [_bad(blk.group(0), "malformed DSML invoke")]
    calls = []
    for inv in invokes:
        name, params = inv.group(1), inv.group(2)
        args: dict[str, Any] = {}
        try:
            if not name.strip() or _DSML_PARAM.sub("", params).strip():
                raise ValueError("empty name or text outside parameters")
            for p in _DSML_PARAM.finditer(params):
                key, is_str, val = p.groups()
                if not key.strip() or key in args:
                    raise ValueError(f"empty or duplicate parameter {key!r}")
                args[key] = val if is_str == "true" else json.loads(val)
        except (ValueError, json.JSONDecodeError) as e:
            calls.append(_bad(inv.group(0), f"invalid DSML: {e}"))
            continue
        calls.append(_call(name, args))
    return text[: blk.start()].strip(), calls


def parse_glm(text: str) -> tuple[str, list[dict[str, Any]]]:
    calls = []
    for blk in _QWEN_BLOCK.finditer(text):
        m = _GLM_TOOL_CALL_RE.match(blk.group(1))
        if not m:
            calls.append(_bad(blk.group(1), "malformed GLM tool call"))
            continue
        body, pos, args, ok = m.group("body"), 0, {}, True
        for p in _GLM_ARG_PAIR_RE.finditer(body):
            if body[pos:p.start()].strip() or not p.group("key").strip():
                ok = False
                break
            args[p.group("key").strip()] = _value(p.group("value"))
            pos = p.end()
        calls.append(_call(m.group("name"), args) if ok and not body[pos:].strip() else _bad(blk.group(1), "content between GLM arguments"))
    return text.split("<tool_call>", 1)[0].strip(), calls


# Kimi K2's native tool-call section (pattern: the model card's tool-calling guide). A Kimi model told the text protocol
# still answers in its own format once a conversation is under way (swarm smoke, 2026-10-23: its third reply was a
# native section, read as a final answer, so the session "finished" without writing anything): the text protocol reads it too
_KIMI_SECTION = "<|tool_calls_section_begin|>"
_KIMI_CALL = re.compile(r"<\|tool_call_begin\|>\s*(?:functions\.)?(?P<name>[\w\-\.]+?)(?::\d+)?\s*<\|tool_call_argument_begin\|>(?P<args>.*?)<\|tool_call_end\|>",
                        re.DOTALL)


def parse_text(text: str) -> tuple[str, list[dict[str, Any]]]:
    """The text protocol: Hermes JSON blocks; failing those, Kimi's native section (an unterminated call is an
    unparseable call the World answers with an error, never a silent final answer)."""
    content, calls = _parse_hermes(text)
    if calls or "<|tool_call_begin|>" not in text:
        return content, calls
    for m in _KIMI_CALL.finditer(text):
        raw = m.group("args").strip()
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            calls.append(_bad(raw, "tool call is not JSON"))
            continue
        calls.append(_call(m.group("name"), _arguments(d)))
    tail = text.rsplit("<|tool_call_end|>", 1)[-1] if calls else text
    if "<|tool_call_begin|>" in tail:
        calls.append(_bad(tail[tail.index("<|tool_call_begin|>"):], "unterminated tool call"))
    head = text.split(_KIMI_SECTION, 1)[0].split("<|tool_call_begin|>", 1)[0]
    return head.strip(), calls


PARSERS = {"qwen_xml": parse_qwen_xml, "dsml": parse_dsml, "glm": parse_glm, "hermes": _parse_hermes, "text": parse_text}

TEXT_PROTOCOL = ("# Tools\n\nYou can call these tools:\n{tools}\n\nTo call a tool, reply with one block per call and nothing after it:\n"
                 "<tool_call>{{\"name\": \"<tool name>\", \"arguments\": {{<arguments as a JSON object>}}}}</tool_call>\n"
                 "Each result comes back in the next user message. Reply without a tool call when the work is done.")


def split_thinking(text: str) -> tuple[str | None, str]:
    """(reasoning, answer) of a completion that may open with a ``<think>`` block (or close one the prompt opened), or of
    a harmony completion (gpt-oss: the ``final`` channel is the answer, earlier channels are reasoning), or of an
    Inkling completion (a ``<|content_thinking|>`` message, then ``<|message_model|><|content_text|>`` with the answer)."""
    if "<|content_text|>" in text and "<|content_thinking|>" in text:
        head, _, tail = text.rpartition("<|content_text|>")
        return head.replace("<|content_thinking|>", "").split("<|end_message|>", 1)[0], tail.split("<|end_message|>", 1)[0]
    if "<|channel|>final<|message|>" in text:
        head, _, tail = text.rpartition("<|channel|>final<|message|>")
        return head, tail.split("<|end|>", 1)[0]
    if "</think>" not in text:
        return None, text
    m = _THINK.match(text)
    if not m:
        return None, text
    return m.group("reasoning").strip(), text[m.end():]


class TinkerChatCodec(HFChatCodec):
    """``HFChatCodec`` for a Tinker model (module doc). ``enable_thinking`` is recorded in ``settings()``."""

    def __init__(self, tokenizer: Any, model: TinkerModel, *, enable_thinking: bool = False, template_kwargs: Mapping[str, Any] | None = None):
        self.model, self.enable_thinking = model, bool(enable_thinking)
        self.tool_format = model.tool_format
        if self.tool_format not in PARSERS:
            raise ValueError(f"{model.name}: unknown tool format {self.tool_format!r}")
        kw = dict(template_kwargs or {})
        if model.thinking_kw:
            kw.setdefault(model.thinking_kw, self.enable_thinking)
        elif not self.enable_thinking:
            for k, v in model.off_kwargs:  # no on/off switch (GLM-5.3 always thinks): the lowest effort stands in, recorded
                kw.setdefault(k, v)
        self.text_protocol = self.tool_format == "text"
        eos = None
        if model.stop_tokens:  # a template with no per-message end token: the next role tokens end a turn
            eos = [int(tokenizer.convert_tokens_to_ids(t)) for t in model.stop_tokens]
            if any(i is None or i == getattr(tokenizer, "unk_token_id", -1) for i in eos):
                raise ValueError(f"{model.name}: a stop token of {model.stop_tokens} is not in the vocabulary")
        self.turn_end_ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in model.turn_end_tokens]
        super().__init__(tokenizer, template_kwargs=kw, eos_ids=eos)

    def suffix(self, new_messages, tools) -> list[int]:
        """``HFChatCodec.suffix``, framing after a history turn-end that does not stop sampling (gpt-oss ``<|end|>``)."""
        if not self.turn_end_ids:
            return super().suffix(new_messages, tools)
        saved = self.eos_ids
        self.eos_ids = [*saved, *self.turn_end_ids]
        try:
            return super().suffix(new_messages, tools)
        finally:
            self.eos_ids = saved

    def settings(self) -> dict[str, Any]:
        return {"model": self.model.name, "tool_format": self.tool_format, "text_protocol": self.text_protocol, "enable_thinking": self.enable_thinking,
                "template_kwargs": dict(self.kw), "eos_ids": list(self.eos_ids)}

    # ---- rendering ----
    @staticmethod
    def _norm_calls(m: dict[str, Any]) -> dict[str, Any]:
        if not m.get("tool_calls"):
            return m
        out = []
        for c in m["tool_calls"]:
            f = dict(c.get("function") or {})
            a = f.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a or "{}")
                except json.JSONDecodeError:
                    a = {"arguments": a}
                f["arguments"] = a if isinstance(a, dict) else {"arguments": a}
            out.append({**c, "function": f})
        return {**m, "tool_calls": out}

    def _text_protocol(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "tool":
                out.append({"role": "user", "content": f"<tool_response>\n{m.get('content') or ''}\n</tool_response>"})
            elif m.get("tool_calls"):
                calls = "".join(f"<tool_call>{json.dumps({'name': c['function']['name'], 'arguments': c['function']['arguments']})}</tool_call>"
                                for c in self._norm_calls(m)["tool_calls"])
                out.append({k: v for k, v in m.items() if k != "tool_calls"} | {"content": (m.get("content") or "") + calls})
            else:
                out.append(m)
        if tools:
            decl = TEXT_PROTOCOL.format(tools="\n".join(json.dumps(t.get("function", t), sort_keys=True) for t in tools))
            if out and out[0].get("role") == "system":
                out[0] = {**out[0], "content": f"{decl}\n\n{out[0].get('content') or ''}"}
            else:
                out.insert(0, {"role": "system", "content": decl})
        return out

    def _apply_raw(self, messages, tools, *, gen: bool) -> list[int]:
        msgs = [self._norm_calls(dict(m)) for m in messages]
        if getattr(self, "text_protocol", False):
            return super()._apply_raw(self._text_protocol(msgs, tools), None, gen=gen)
        return super()._apply_raw(msgs, tools, gen=gen)

    # ---- parsing ----
    def decode(self, completion_ids: list[int]) -> str:
        ids = list(completion_ids)
        while ids and ids[-1] in self.eos_ids:
            ids.pop()
        return self.tok.decode(ids, skip_special_tokens=False)

    def parse(self, completion_ids: list[int]) -> tuple[str, list[dict[str, Any]]]:
        text = self.decode(completion_ids)
        _, answer = split_thinking(text)
        content, calls = PARSERS[self.tool_format](answer)
        for s in (getattr(self.tok, "all_special_tokens", None) or []):  # stray specials (a pad, an eos the template did not name)
            if s and s in content and not s.startswith("<｜DSML"):
                content = content.replace(s, "")
        return content.strip(), calls


def load_tokenizer(model: TinkerModel) -> Any:
    """The model's tokenizer from the Hugging Face hub or cache (``TinkerModel.base``; tokenizer files only)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model.base)
