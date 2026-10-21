"""M2 R-2 (audit 2026-10-15): a budget hold prices its own call, and pressure from calls in flight waits instead of
failing. Every hold was floored at the dearest call so far, so after one 0.20 USD call 5 of 16 cheap concurrent calls
raised ``credits_exhausted`` (which stops the run) with 91% of the budget unused (probes v3_ss1, ss02)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from crucible.actors.llm import BudgetExceeded, LLMClient
from crucible.core.config import Config, LLMProvider


class Completions:
    """Answers after ``delay_s``; usage from the prompt's characters (chars/4) and 50 output tokens; ``cost`` is the
    provider-reported cost (None: priced from config)."""

    def __init__(self, *, cost: float | None = None, delay_s: float = 0.05):
        self.cost, self.delay_s, self.calls = cost, delay_s, 0

    async def create(self, **kw):
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        pt = sum(len(str(m.get("content", ""))) for m in kw["messages"]) // 4
        msg = NS(content="ok", tool_calls=None, refusal=None, model_dump=lambda exclude_none=True: {"role": "assistant", "content": "ok"})
        usage = NS(prompt_tokens=pt, completion_tokens=50, cost=self.cost, completion_tokens_details=None, prompt_tokens_details=None, model_extra={})
        return NS(choices=[NS(message=msg, finish_reason="stop", model_extra={}, logprobs=None)], usage=usage, model=kw["model"], id="r", model_extra={})


def client(tmp_path, budget: float, price: list[float], completions: Completions) -> LLMClient:
    cfg = Config(llm_providers={"or": LLMProvider(base_url="https://openrouter.ai/api/v1", api_key="env:OPENROUTER_API_KEY")},
                 prices={"vendor/m": price})
    c = LLMClient("or:vendor/m", config=cfg, budget_usd=budget, cache_mode="off", cache_dir=tmp_path)
    c._client = NS(chat=NS(completions=completions))
    return c


async def test_one_dear_call_no_longer_inflates_every_later_hold(tmp_path):
    c = client(tmp_path, 2.5, [2.0, 10.0], Completions())
    await c.complete([{"role": "user", "content": "x" * 400_000}], max_tokens=1200)  # ~100k prompt tokens: 0.2005 USD
    out = await asyncio.gather(*(c.complete([{"role": "user", "content": f"{i} " + "y" * 4000}], max_tokens=1200) for i in range(16)),
                               return_exceptions=True)
    assert not [o for o in out if isinstance(o, BaseException)]  # was: 5 of 16 credits_exhausted, the run stopped
    assert c.spend.calls == 17 and c.spend.usd <= 0.45 and c._reserved == pytest.approx(0.0)


async def test_pressure_from_calls_in_flight_waits_for_them(tmp_path):
    c = client(tmp_path, 0.004, [0.1, 0.5], Completions(cost=0.00005))  # holds ~0.0004 each: 10 fit at once
    out = await asyncio.gather(*(c.complete([{"role": "user", "content": f"hello {i}"}], max_tokens=800) for i in range(30)),
                               return_exceptions=True)
    assert not [o for o in out if isinstance(o, BaseException)]  # was: 21 of 30 refused by holds alone
    assert c.spend.calls == 30 and c.spend.budget_waits > 0 and c.spend.usd <= 0.004


async def test_true_exhaustion_raises_at_once_even_with_calls_in_flight(tmp_path):
    comp = Completions(cost=0.0003, delay_s=0.2)
    c = client(tmp_path, 0.001, [0.1, 0.5], comp)
    c.spend.usd = 0.0005
    slow = asyncio.create_task(c.complete([{"role": "user", "content": "slow"}], max_tokens=800))  # holds 0.0004
    await asyncio.sleep(0.01)
    c.spend.usd = 0.0007  # booked meanwhile by another call
    with pytest.raises(BudgetExceeded):  # 0.0007 spent + this call's 0.0004 cannot fit whatever settles: no wait
        await asyncio.wait_for(c.complete([{"role": "user", "content": "now"}], max_tokens=800), 0.1)
    await slow
    assert comp.calls == 1 and c.spend.budget_waits == 0 and c._reserved == pytest.approx(0.0)


async def test_a_provider_dearer_than_its_price_scales_later_holds(tmp_path):
    comp = Completions(cost=0.004)  # ten times the configured price's worst case
    c = client(tmp_path, 0.005, [0.1, 0.5], comp)
    await c.complete([{"role": "user", "content": "first"}], max_tokens=800)
    assert c._overrun > 9
    out = await asyncio.gather(*(c.complete([{"role": "user", "content": f"q{i}"}], max_tokens=800) for i in range(5)), return_exceptions=True)
    assert all(isinstance(o, BudgetExceeded) for o in out) and comp.calls == 1 and c.spend.usd <= 0.005  # AUD-D12 stays fixed
