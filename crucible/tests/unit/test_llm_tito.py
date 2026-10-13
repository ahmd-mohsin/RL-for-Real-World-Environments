"""``LLMProvider.tito``: exact token-in-token-out capture through a self-hosted engine. A chat API re-renders history
through its tool parser + template, which is lossy (lab M10: Qwen3 + hermes turned ``open.\\n\\n<tool_call>`` into
``open.\\n\\n\\n<tool_call>``, a prefix break on 23 of 24 calls). In tito mode the client renders with the model's own
template, extends each served conversation token for token, and parses tool calls itself."""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

MODEL = "Qwen/Qwen3-0.6B"


def _tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(MODEL)
    except OSError:
        pytest.skip(f"{MODEL} tokenizer not cached")


async def test_tito_interleaved_conversations_extend_exactly(tmp_path, monkeypatch):
    from crucible.actors.llm import LLMClient
    from crucible.core.config import Config, LLMProvider
    from crucible.train.engine import GenResult
    from crucible.train.vllm_http import VLLMServerEngine

    tok = _tok()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    replies = {  # what the model "samples": text before the call ending in ".\n\n" (one token), as in the lab
        "A": 'Let me check the open tickets.\n\n<tool_call>\n{"name": "list_tickets", "arguments": {"status": "open"}}\n</tool_call>',
        "B": 'Looking up order 7.\n\n<tool_call>\n{"name": "get_order", "arguments": {"id": 7}}\n</tool_call>',
    }
    prompts: list[list[int]] = []

    async def fake_generate(self, req):
        prompts.append(list(req.prompt_ids))
        text = tok.decode(req.prompt_ids)
        key = "A" if "tickets" in text.split("<|im_start|>user")[1] else "B"
        ids = tok.encode(replies[key], add_special_tokens=False) + [eos]
        return GenResult(ids, [-0.5] * len(ids), 0, "stop", None, len(req.prompt_ids))

    monkeypatch.setattr(VLLMServerEngine, "generate", fake_generate)
    cfg = Config(llm_providers={"lab": LLMProvider(base_url="http://127.0.0.1:1/v1", tito=True, policy_version=3,
                                                   template_kwargs={"enable_thinking": False})},
                 models={"p": f"lab:{MODEL}"}, data_dir=str(tmp_path))
    llm = LLMClient("p", config=cfg, cache_mode="off")
    tools = [{"type": "function", "function": {"name": n, "description": "d", "parameters": {"type": "object", "properties": {}}}}
             for n in ("list_tickets", "get_order")]
    conv = {"A": [{"role": "user", "content": "Work the open tickets."}], "B": [{"role": "user", "content": "Where is order 7?"}]}
    first = {}
    for k in ("A", "B"):  # interleaved: A1, B1, A2, B2 through one client
        r = await llm.complete(conv[k], tools=tools, temperature=1.0, max_tokens=64)
        assert r.tool_calls and r.tool_calls[0]["name"] == ("list_tickets" if k == "A" else "get_order")
        assert r.tokens["policy_version"] == 3 and len(r.tokens["logprobs"]) == len(r.tokens["completion_token_ids"])
        first[k] = r.tokens["prompt_token_ids"] + r.tokens["completion_token_ids"]
        c = r.tool_calls[0]
        # the assistant turn replays ``raw_message`` as ``LLMToolPolicy`` does: its reply nonce names the served conversation
        conv[k] += [{**r.raw_message, "role": "assistant", "content": r.text, "tool_calls": [{"id": c["id"], "type": "function",
                                                                                               "function": {"name": c["name"], "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": c["id"], "content": "result"}]
    for k in ("A", "B"):
        r = await llm.complete(conv[k], tools=tools, temperature=1.0, max_tokens=64)
        assert r.tokens["prompt_token_ids"][: len(first[k])] == first[k], k  # extends what was sampled, token for token
    assert llm._tito.prefix_breaks == 0 and llm.spend.calls == 4
    # the chat-API path would have re-rendered history: the template does not reproduce the sampled tokens
    from crucible.train.chat import HFChatCodec

    codec = HFChatCodec(tok, template_kwargs={"enable_thinking": False})
    rerendered = codec.render(conv["A"], tools)
    assert rerendered[: len(first["A"])] != first["A"]


def test_engine_llm_compaction_starts_a_new_segment():
    import asyncio

    from crucible.train.chat import ByteChatCodec
    from crucible.train.engine import GenResult
    from crucible.train.world_env import EngineLLM

    codec = ByteChatCodec()

    async def gen(prompt, n, t):
        return GenResult(list(b"ok") + [256], [-0.1] * 3, 0, "stop")

    llm = EngineLLM(codec, gen)

    async def run():
        m = [{"role": "user", "content": "hi"}]
        await llm.complete(m)
        m2 = m + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
        await llm.complete(m2)
        assert llm.prefix_breaks == 0
        rewritten = [{"role": "user", "content": "summary"}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}]
        await llm.complete(rewritten)
        assert llm.prefix_breaks == 1

    asyncio.run(run())
