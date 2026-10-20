"""$0 probe: for each Tinker model, our codec on its own HF chat template (render with tools, a history with a tool call,
the TITO suffix, thinking behaviour, EOS ids) and whether our parser reads back the template's own rendering of a call."""
import json
import sys
import traceback

from crucible.train.bridges.tinker.codec import TinkerChatCodec, load_tokenizer
from crucible.train.bridges.tinker.models import resolve

MODELS = sys.argv[1:] or ["Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.5-397B-A17B", "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144",
                          "zai-org/GLM-5.3-Flash:peft:262144", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"]
TOOLS = [{"type": "function", "function": {"name": "repo__shell", "description": "Run a shell command", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
out = {}
for name in MODELS:
    r = {}
    try:
        m = resolve(name)
        tok = load_tokenizer(m)
        c = TinkerChatCodec(tok, m, enable_thinking=False)
        r["settings"] = c.settings()
        r["eos"] = [tok.decode([i]) for i in c.eos_ids]
        hist = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "repo__shell", "arguments": json.dumps({"command": "ls -la"})}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "a.py"}]
        gen = tok.decode(c.render(hist[:2], TOOLS))
        r["gen_prompt_tail"] = gen[-80:]
        full = tok.decode(c.render(hist, TOOLS))
        r["history_render_tail"] = full[-400:]
        r["suffix"] = tok.decode(c.suffix([{"role": "tool", "tool_call_id": "c1", "content": "out"}], TOOLS))
        try:
            r["thinking"] = c.thinking_behaviour()
        except Exception as e:  # noqa: BLE001
            r["thinking"] = f"error {type(e).__name__}: {e}"
        # the template's own rendering of the call, cut out between the prompt for the turn and the next turn
        pre = tok.decode(c.render(hist[:2], TOOLS))
        k = full.find(pre[-40:]) if pre[-40:] in full else -1
        seg = full[k + 40:] if k >= 0 else full
        ids = tok.encode(seg, add_special_tokens=False)
        cut = next((i for i, t in enumerate(ids) if t in c.eos_ids), len(ids))
        text, calls = c.parse(ids[:cut] + c.eos_ids[:1])
        r["roundtrip"] = {"text": text[:120], "calls": calls}
        r["ok"] = bool(calls) and calls[0]["name"] == "repo__shell" and calls[0]["arguments"] == {"command": "ls -la"}
    except Exception as e:  # noqa: BLE001
        r["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        r["trace"] = traceback.format_exc()[-1500:]
    out[name] = r
print(json.dumps(out, indent=1, default=str))
