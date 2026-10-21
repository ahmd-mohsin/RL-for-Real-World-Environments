"""M2 R-24 (audit 2026-10-15): a paid call that failed after it may have reached the model is booked as uncertain spend
and sent again at most once. A read timeout was re-sent and booked nothing, so the hard budget and the run's
uncertain-spend rule never saw the dearest calls (probe ss04: 2 requests sent, 0.0 booked)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import httpx
import openai
import pytest

import crucible.actors.llm as llm_mod
from crucible.actors.llm import LLMClient, LLMError, RequestRejected, _approx_tokens
from crucible.core.config import Config, LLMProvider

REQ = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
MESSAGES = [{"role": "user", "content": "x" * 4000}]


def read_timeout():
    try:
        raise httpx.ReadTimeout("read timed out", request=REQ)
    except httpx.ReadTimeout as e:
        raise openai.APITimeoutError(request=REQ) from e


def refused():
    try:
        raise httpx.ConnectError("connection refused", request=REQ)
    except httpx.ConnectError as e:
        raise openai.APIConnectionError(request=REQ) from e


def server_error():
    raise openai.InternalServerError("upstream died", response=httpx.Response(502, request=REQ), body=None)


def bad_request():
    raise openai.BadRequestError("unknown parameter", response=httpx.Response(400, request=REQ), body=None)


def ok():
    msg = NS(content="ok", tool_calls=None, refusal=None, model_dump=lambda exclude_none=True: {"role": "assistant", "content": "ok"})
    usage = NS(prompt_tokens=1000, completion_tokens=50, cost=0.001, completion_tokens_details=None, prompt_tokens_details=None, model_extra={})
    return NS(choices=[NS(message=msg, finish_reason="stop", model_extra={}, logprobs=None)], usage=usage, model="vendor/m", id="r", model_extra={})


class Completions:
    def __init__(self, *outcomes):
        self.outcomes, self.sent = list(outcomes), 0

    async def create(self, **kw):
        self.sent += 1
        return (self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0])()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod, "_retry_delay", lambda e, attempt, cap=60.0: 0.0)

    def make(*outcomes) -> tuple[LLMClient, Completions]:
        cfg = Config(llm_providers={"or": LLMProvider(base_url="https://openrouter.ai/api/v1", api_key="env:OPENROUTER_API_KEY")},
                     prices={"vendor/m": [1.0, 2.0]})
        c = LLMClient("or:vendor/m", config=cfg, budget_usd=1.0, cache_mode="off", cache_dir=tmp_path, max_retries=3)
        comp = Completions(*outcomes)
        c._client = NS(chat=NS(completions=comp))
        return c, comp

    return make


HOLD = _approx_tokens(MESSAGES) / 1e6 * 1.0 + 4000 / 1e6 * 2.0


async def test_a_call_lost_after_it_was_sent_is_booked_and_sent_once_more(client):
    c, comp = client(read_timeout)
    with pytest.raises(LLMError):
        await c.complete(MESSAGES, max_tokens=4000)
    assert comp.sent == 2  # was 2 sends and 0.0 booked; max_retries=3 would have sent 3
    assert c.spend.usd == pytest.approx(2 * HOLD) and c.spend.uncertain_usd == pytest.approx(2 * HOLD) and c.spend.failed_calls == 2
    assert c._reserved == pytest.approx(0.0)


async def test_a_lost_call_then_a_success_books_both(client):
    c, comp = client(server_error, ok)
    r = await c.complete(MESSAGES, max_tokens=4000)
    assert r.text == "ok" and comp.sent == 2
    assert c.spend.usd == pytest.approx(HOLD + 0.001) and c.spend.uncertain_usd == pytest.approx(HOLD) and c.spend.calls == 1


@pytest.mark.parametrize(("outcome", "error", "sent"), [(refused, LLMError, 3), (bad_request, RequestRejected, 1)])
async def test_a_call_that_never_ran_books_nothing(client, outcome, error, sent):
    c, comp = client(outcome)
    with pytest.raises(error):
        await c.complete(MESSAGES, max_tokens=4000)
    assert comp.sent == sent and c.spend.usd == 0.0 and c.spend.uncertain_usd == 0.0 and c.spend.failed_calls == 0
