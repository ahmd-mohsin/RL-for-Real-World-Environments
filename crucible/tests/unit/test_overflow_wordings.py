"""M2 R-15 (audit 2026-10-15): one overflow wording table for every model-calling path. The nine markers caught 2 of 8
provider wordings on the hosted and RLM paths and 1 on the token-in-token-out path (probe v2_pvc4); a missed overflow is
masked as an outage (or retried into the same overflow) instead of being scored."""

from __future__ import annotations

from types import SimpleNamespace as NS

import httpx
import pytest

from crucible.actors.llm import CONTEXT_OVERFLOW_MARKERS, NON_OVERFLOW_PATTERNS, OVERFLOW_PATTERNS, ContextOverflow, LLMClient, LLMError, is_context_overflow, is_overflow_text
from crucible.actors.rlm.context import is_overflow
from crucible.core.config import Config, LLMProvider

SAMPLES = [  # one per OVERFLOW_PATTERNS row, in order
    "prompt is too long: 213462 tokens > 200000 maximum",
    "request_too_large",
    "Input is too long for requested model.",
    "Your input exceeds the context window of this model.",
    "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575).",
    "This model's maximum prompt length is 131072 but the request contains 537812 tokens.",
    "Please reduce the length of the messages or completion.",
    "This endpoint's maximum context length is 128000 tokens. However, you requested about 130000 tokens.",
    "litellm.ContextWindowExceededError: the input exceeds the model's maximum context length",
    "prompt token count of 130000 exceeds the limit of 128000",
    "the request exceeds the available context size, try increasing it",
    "the model is loaded with context length of 8192 which is greater than the context length",
    "invalid params, context window exceeds limit",
    "Your request exceeded model token limit: 262144",
    "Prompt contains 140000 tokens, too large for model with 131072 maximum context length",
    "model_context_window_exceeded",
    "the combined input and output tokens exceed the model limit",
    "This model accepts at most 1048576 combined tokens",
    "Please reduce the input length or requested output length.",
    "prompt too long; exceeded max context length by 412 tokens",
    "context_length_exceeded",
    "Request has too many tokens",
    "token limit exceeded",
    "413 (no body)",
    "Requested tokens exceed the maximum context length",
    "413 Request Entity Too Large",
    "The prompt exceeds the maximum number of tokens allowed",
    "Error: maximum prompt length is exceeded",
    "The input (140012 tokens) is longer than the model's context length (131072 tokens).",
    "Input prompt (2048 tokens) is too long and exceeds limit of 1024",
]
NOT_OVERFLOW = ["Throttling error: Too many tokens, please wait before trying again.", "Rate limit reached: too many tokens per minute",
                "Too Many Requests: token limit exceeded for this minute", "unknown parameter: temperature"]


def test_every_pattern_has_its_sample_and_the_old_markers_are_covered():
    assert len(SAMPLES) == len(OVERFLOW_PATTERNS) and len(NON_OVERFLOW_PATTERNS) == 3
    assert all(is_overflow_text(m) for m in CONTEXT_OVERFLOW_MARKERS)


@pytest.mark.parametrize(("i", "text"), list(enumerate(SAMPLES)))
def test_each_wording_is_an_overflow(i, text):
    assert OVERFLOW_PATTERNS[i].search(text) and is_overflow_text(text)


@pytest.mark.parametrize("text", NOT_OVERFLOW)
def test_rate_limits_worded_like_an_overflow_are_not(text):
    assert not is_overflow_text(text)


class Http(Exception):
    def __init__(self, text: str, status: int):
        super().__init__(text)
        self.status_code = status


PROBE = [  # v2_pvc4: (status, wording, overflow?)
    (400, "The input (140012 tokens) is longer than the model's context length (131072 tokens).", True),  # SGLang
    (400, "This model's maximum context length is 32768 tokens. However, you requested 40000 tokens.", True),  # vLLM
    (400, "the request exceeds the available context size, try increasing it", True),  # llama.cpp
    (400, "the model is loaded with context length of 8192 which is greater than the context length", True),  # LM Studio
    (400, "prompt too long; exceeded max context length by 412 tokens", True),  # Ollama
    (400, "Input is too long for requested model.", True),  # Bedrock
    (400, "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).", True),  # Gemini
    (413, "413 (no body)", True),  # Cerebras
    (429, "Rate limit reached: too many tokens per minute", False),
]


@pytest.mark.parametrize(("status", "text", "overflow"), PROBE)
async def test_the_hosted_rlm_and_engine_paths_decide_alike(tmp_path, status, text, overflow):
    assert is_context_overflow(Http(text, status)) is overflow  # hosted: was 2 of 8
    assert is_overflow(Http(text, status)) is overflow  # RLM: was 2 of 8
    cfg = Config(llm_providers={"lab": LLMProvider(base_url="http://127.0.0.1:1/v1", tito=True)})
    c = LLMClient("lab:org/m", config=cfg, cache_mode="off", cache_dir=tmp_path)
    req = httpx.Request("POST", "http://127.0.0.1:1/v1/completions")

    async def refuse(*a, **kw):
        raise httpx.HTTPStatusError(text, request=req, response=httpx.Response(status, text=text, request=req))

    c._tito = NS(complete=refuse)
    with pytest.raises(LLMError) as got:  # token-in-token-out: was 1 of 8
        await c.complete([{"role": "user", "content": "q"}])
    assert isinstance(got.value, ContextOverflow) is overflow
