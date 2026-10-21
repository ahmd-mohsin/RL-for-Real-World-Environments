"""M2 R-14 (audit 2026-10-15): the compaction budget is the model's declared or discovered context window. It was a fixed
128,000 tokens, so on a 32,768-token engine the default ContextConfig never compacted and the episode ended
context_overflow after 7 calls (probe hb01)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from crucible.actors.llm import ContextOverflow, LLMClient, LLMResponse
from crucible.actors.policy import COMPACT, ContextConfig, LLMToolPolicy, approx_tokens
from crucible.core.config import Config, LLMProvider
from crucible.core.schema.actions import Done, Observation, ToolCall, ToolResult

WINDOW = 32_768


class Stub32k:
    """A 32k engine: refuses a prompt over the window; every turn reads a ~6k-token file. ``window`` is what its client
    knows of the window (None: nothing, as before R-14)."""

    model = "stub-32k"

    def __init__(self, window: int | None):
        self.window, self.summaries = window, 0

    async def context_window(self):
        return self.window

    async def complete(self, messages, *, tools=None, **kw):
        n = approx_tokens(messages)
        if n > WINDOW:
            raise ContextOverflow(f"stub: maximum context length is {WINDOW} tokens; request has {n}")
        usage = {"prompt_tokens": n, "completion_tokens": 20}
        if messages[-1].get("content") == COMPACT:
            self.summaries += 1
            return LLMResponse(text="summary: read files 1..k, nothing found yet", usage=usage, raw_finish="stop")
        return LLMResponse(text="", tool_calls=[{"id": f"c{n}", "name": "fs__read", "arguments": {"path": "f"}}], usage=usage, raw_finish="tool_calls")


async def _run(llm: Stub32k, steps: int = 30) -> LLMToolPolicy:
    pol = LLMToolPolicy("agent", llm, context=ContextConfig())  # type: ignore[arg-type]
    obs = Observation(actor="agent", step=0, instruction="Find the config bug.")
    for step in range(1, steps):
        a = await pol.act(obs)
        if isinstance(a, Done):
            break
        assert isinstance(a, ToolCall)
        obs = Observation(actor="agent", step=step, tool_results=[ToolResult(tool_call_id=a.tool_call_id, function_name=a.function_name, ok=True,
                                                                               content="x" * 24_000)])
    return pol


async def test_the_default_budget_is_the_models_window():
    pol = await _run(Stub32k(WINDOW))
    assert pol.stop is None and len(pol.segments) >= 5  # compacted all along
    blind = await _run(Stub32k(None))
    assert blind.stop == "context_overflow" and blind.segments == []  # unknown window: 128,000 as before


def _client(tmp_path, *, base_url: str, windows: dict[str, int] | None = None, cache_mode: str = "off", models=None) -> LLMClient:
    cfg = Config(llm_providers={"p": LLMProvider(base_url=base_url, api_key="EMPTY")}, context_windows=windows or {})
    c = LLMClient("p:org/model-a", config=cfg, cache_mode=cache_mode, cache_dir=tmp_path)
    listed: list[int] = []

    async def list_models():
        listed.append(1)
        if isinstance(models, Exception):
            raise models
        return NS(data=models or [])

    c._client = NS(models=NS(list=list_models))
    c.listed = listed  # type: ignore[attr-defined]
    return c


async def test_a_self_hosted_engine_is_asked_once_and_config_wins(tmp_path):
    cards = [NS(id="org/other", max_model_len=4096), NS(id="org/model-a", object="model", max_model_len=WINDOW)]
    c = _client(tmp_path, base_url="http://127.0.0.1:8000/v1", models=cards)
    assert await c.context_window() == WINDOW and await c.context_window() == WINDOW
    assert (c.window_source, len(c.listed)) == ("models", 1)
    c = _client(tmp_path, base_url="http://127.0.0.1:8000/v1", windows={"org/model-a": 8192}, models=cards)
    assert (await c.context_window(), c.window_source, c.listed) == (8192, "config", [])
    c = _client(tmp_path, base_url="https://openrouter.ai/api/v1", windows={"org/model-a": 200_000})
    assert (await c.context_window(), c.window_source) == (200_000, "config")  # a hosted model: from config only


async def test_an_unknown_window_never_fails_and_never_goes_online_from_a_cache(tmp_path):
    clients = [_client(tmp_path, base_url="https://openrouter.ai/api/v1", models=[NS(id="org/model-a", context_length=WINDOW)]),
               _client(tmp_path, base_url="http://127.0.0.1:8000/v1", cache_mode="replay", models=[NS(id="org/model-a", max_model_len=WINDOW)]),
               _client(tmp_path, base_url="http://127.0.0.1:8000/v1", models=ConnectionError("refused")),
               _client(tmp_path, base_url="http://127.0.0.1:8000/v1", models=[NS(id="org/model-a")])]  # mlx-lm: no window field
    for c in clients:
        assert (await c.context_window(), c.window_source) == (None, "unknown")
    assert [len(c.listed) for c in clients] == [0, 0, 1, 1]  # hosted and replayed clients never ask


def test_the_rlm_keeps_its_budget_when_context_config_leaves_it_unset():
    from crucible.actors.rlm.policy import _config

    cfg = _config(None, {"context": ContextConfig()})
    assert cfg.window() == 128_000 and cfg.mode() != "none"  # None must not read as "compaction off"


def test_config_refuses_a_window_that_is_not_positive():
    with pytest.raises(ValueError, match="context_windows"):
        Config(context_windows={"m": 0})
