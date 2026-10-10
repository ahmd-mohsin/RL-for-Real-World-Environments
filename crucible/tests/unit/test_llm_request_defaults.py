"""Self-hosted engines need token capture requested on every call: a provider's ``request_defaults`` (e.g. vLLM
``logprobs`` + ``return_token_ids``) reach the request, and the response's token record is parsed (train mode)."""

from __future__ import annotations

import asyncio

from openai.types.chat import ChatCompletion

from crucible.actors.llm import LLMClient
from crucible.core.config import Config


def test_provider_request_defaults_and_token_record(tmp_path):
    cfg = Config.model_validate({"llm_providers": {"lab": {"base_url": "http://127.0.0.1:8000/v1", "api_key": "EMPTY",
                                                           "request_defaults": {"logprobs": True, "extra_body": {"return_token_ids": True}}}},
                                 "models": {"policy-default": "lab:Qwen/Qwen3-4B-Instruct-2507"}, "data_dir": str(tmp_path)})
    llm = LLMClient("policy-default", config=cfg, cache_mode="off")
    seen = {}

    class Completions:
        async def create(self, **kw):
            seen.update(kw)
            return ChatCompletion.model_validate({
                "id": "x", "object": "chat.completion", "created": 0, "model": kw["model"], "prompt_token_ids": [1, 2, 3],
                "choices": [{"index": 0, "finish_reason": "stop", "token_ids": [7, 8],
                             "message": {"role": "assistant", "content": "hi"},
                             "logprobs": {"content": [{"token": "h", "logprob": -0.5, "bytes": None, "top_logprobs": []},
                                                      {"token": "i", "logprob": -0.25, "bytes": None, "top_logprobs": []}]}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    llm._openai = lambda: Client()
    r = asyncio.run(llm.complete([{"role": "user", "content": "hi"}], extra_body={"top_k": 20}))
    assert seen["logprobs"] is True and seen["extra_body"] == {"return_token_ids": True, "top_k": 20}  # defaults merged, call-level wins
    assert r.tokens == {"prompt_token_ids": [1, 2, 3], "completion_token_ids": [7, 8], "logprobs": [-0.5, -0.25], "policy_version": None}
