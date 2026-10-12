"""M2 audit (2026-10-12) regression tests: each case reproduced a defect before its fix (``docs/audit/2026-10-12/M02-actors.md``)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from crucible.actors.grounding import check, is_factual
from crucible.actors.records import RecordStore, tokens
from crucible.core.schema.trace import Record


def R(i: str, text: str, time: str | None = None) -> Record:
    return Record(id=i, text=text, time=time, source="s", role="reviewer")


STORE = RecordStore([R("r1", "The refund of 100 USD was approved on 2024-05-01 for order INV-1042.", "2024-05-02T00:00:00Z"),
                     R("r2", "Shipping is free for orders over 50 USD.", "2024-01-01T00:00:00Z"),
                     R("r3", "The warehouse ships on Mondays.", "2024-01-01T00:00:00Z")])
NOW = "2024-06-01T00:00:00Z"


@pytest.mark.parametrize("claim,why", [
    ("The refund of 10 USD was approved [R:r1].", "facts not in cited records"),  # D1: "10" matched inside "100"
    ("The refund was approved on 2024-01-05 [R:r1].", "facts not in cited records"),  # D2: a swapped date matched digit by digit
    ("Sure, the refund of 900 USD was approved.", "no citation"),  # D3: a greeting prefix made the claim "phatic"
    ("Thanks, your order ships today with express delivery.", "no citation"),
    ("The refund of 100 USD was not approved [R:r1].", "polarity"),  # D4: a negated claim passed on full overlap
    ("Orders over 50 USD ship free on Mondays with a 100 USD refund [R:r1] [R:r2] [R:r3].", "citations"),  # D5: pooled records
])
def test_grounding_gate_rejects_fabrications(claim, why):
    rep = check(claim, STORE, now=NOW)
    assert not rep.ok and why in rep.unsupported[0].reason, rep.sentences


def test_grounding_gate_still_accepts_faithful_claims():
    for claim in ("The refund of 100 USD was approved on 2024-05-01 [R:r1].", "Shipping is free for orders over 50 USD [R:r2].",
                  "Thanks. The warehouse ships on Mondays [R:r3].", "The warehouse ships on Mondays, and shipping is free over 50 USD [R:r3] [R:r2]."):
        rep = check(claim, STORE, now=NOW)
        assert rep.ok, (claim, rep.sentences)
    neg = RecordStore([R("n1", "The refund was not approved because the receipt is missing.", "2024-01-01T00:00:00Z")])
    assert check("The refund was not approved [R:n1].", neg, now=NOW).ok
    assert not check("The refund was approved [R:n1].", neg, now=NOW).ok  # the reverse flip too
    assert not is_factual("Thanks!") and not is_factual("Sure, thank you.") and is_factual("Sure, the refund was 900 USD.")


def test_records_index_every_script_and_hide_untimed_records_in_replays():
    assert tokens("café José naïve") == ["café", "josé", "naïve"]  # D6: ASCII-only tokens ("caf", "jos")
    assert tokens("возврат одобрен") == ["возврат", "одобрен"] and tokens("استرداد") == ["استرداد"]  # were dropped entirely
    assert "退款" in tokens("退款已批准") and "返金" in tokens("返金承認")
    ru = RecordStore([R("x1", "Возврат 100 рублей одобрен.")])
    assert [r.id for r, _ in ru.search("возврат одобрен")] == ["x1"]
    timeless = [R("u1", "the 2025 fix: use a mutex")]
    assert RecordStore(timeless).search("fix mutex", now="2020-01-01T00:00:00Z") == []  # D7: leaked into a 2020 replay
    assert [r.id for r, _ in RecordStore(timeless).search("fix mutex")] == ["u1"]  # no replay clock: visible
    assert [r.id for r, _ in RecordStore(timeless, untimed="visible").search("fix mutex", now="2020-01-01T00:00:00Z")] == ["u1"]
    with pytest.raises(ValueError, match="record 't1'"):  # D8: a timezone-less time crashed the first search, not the load
        RecordStore([R("t1", "x", "2024-05-02T00:00:00")])


async def test_file_queue_answers_only_the_question_asked(tmp_path: Path):
    from crucible.actors.transports import FileQueueTransport

    t = FileQueueTransport(tmp_path, timeout_s=2, poll_s=0.02)
    snap = t.state_dict()

    async def answer(text: str) -> None:
        for _ in range(200):
            qs = sorted(tmp_path.glob("q-*.json"))
            fresh = [q for q in qs if not (tmp_path / q.name.replace("q-", "a-", 1)).exists()]
            if fresh:
                (tmp_path / fresh[0].name.replace("q-", "a-", 1)).write_text(json.dumps({"text": text}))
                return
            await asyncio.sleep(0.02)

    first, _ = await asyncio.gather(t.send("Approve PO 77?"), answer("Yes, PO 77 approved."))
    assert first == "Yes, PO 77 approved."
    t.load_state_dict(snap)  # D9: a World restored to before the question read that answer for a different question
    second, _ = await asyncio.gather(t.send("Approve PO 99 for 1M USD?"), answer("No."))
    assert second == "No."
    other = FileQueueTransport(tmp_path, timeout_s=0.3, poll_s=0.02)  # another episode sharing the directory
    with pytest.raises(TimeoutError):
        await other.send("Unrelated question")


async def test_relay_bot_literal_braces_and_pending_lookups_survive_snapshot():
    from crucible.actors.scripted import RelayActor, Rule
    from crucible.core.schema.actions import Inbound, Message, ToolCall, ToolResult
    from crucible.world.world import ActorTurn

    bot = RelayActor("x", [Rule(r"status (\d+)", reply='JSON: {"order": "{0}", "status": "ok"}'),
                           Rule(r"order (\d+)", tool="db.lookup", args={"id": "{0}"}, reply="Order {0}: {tool_result}")])
    out = await bot.respond(ActorTurn(actor="x", inbox=[Inbound(frm="agent", text="status 7")], tool_results=[], tools=[], time=None))
    assert isinstance(out[0], Message) and out[0].text == 'JSON: {"order": "7", "status": "ok"}'  # D10: KeyError on literal braces
    [call] = await bot.respond(ActorTurn(actor="x", inbox=[Inbound(frm="agent", text="order 42")], tool_results=[], tools=[], time=None))
    assert isinstance(call, ToolCall) and call.arguments == {"id": "42"}
    state = json.loads(json.dumps(bot.state_dict()))  # D11: the pending lookup was not in the state (and held a regex match)
    fresh = RelayActor("x", bot.rules)
    fresh.load_state_dict(state)
    [reply] = await fresh.respond(ActorTurn(actor="x", inbox=[], tool_results=[ToolResult(tool_call_id=call.tool_call_id, function_name="db.lookup", ok=True,
                                                                                            content="shipped")], tools=[], time=None))
    assert reply.text == "Order 42: shipped" and reply.to == "agent"


def _fake(cost: float):
    msg = NS(content="ok", tool_calls=None, model_dump=lambda exclude_none=True: {"role": "assistant", "content": "ok"})
    usage = NS(prompt_tokens=100, completion_tokens=10, cost=cost, completion_tokens_details=None, prompt_tokens_details=None, model_extra={})

    class Fake:
        calls = 0
        chat = property(lambda self: self)
        completions = property(lambda self: self)

        async def create(self, **kw):
            Fake.calls += 1
            await asyncio.sleep(0.02)
            return NS(choices=[NS(message=msg, finish_reason="stop", logprobs=None)], usage=usage, model="fake", id="x")

    return Fake()


async def test_llm_budget_is_hard_under_concurrency(tmp_path: Path):
    from crucible.actors.llm import BudgetExceeded, LLMClient

    c = LLMClient("policy-default", budget_usd=0.005, cache_mode="off", cache_dir=tmp_path)
    c._client = _fake(0.0004)
    out = await asyncio.gather(*(c.complete([{"role": "user", "content": f"q{i}"}]) for i in range(40)), return_exceptions=True)
    assert c.spend.usd <= 0.005 + 1e-9, c.spend.usd  # D12: 40 concurrent calls spent 8x a 0.05 budget
    assert sum(isinstance(o, BudgetExceeded) for o in out) > 0 and c._reserved == pytest.approx(0.0)


async def test_llm_cache_writes_atomically(tmp_path: Path):
    from crucible.actors.llm import LLMClient

    c = LLMClient("policy-default", budget_usd=1, cache_mode="record", cache_dir=tmp_path)
    c._client = _fake(0.0001)
    await c.complete([{"role": "user", "content": "same"}])
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["text"] == "ok" and not list(tmp_path.rglob("*.tmp"))  # D13


def test_simulator_judge_sees_its_own_tool_results():
    import inspect

    from crucible.actors.simulated import JUDGE, SimulatedActor

    src = inspect.getsource(SimulatedActor._judge)
    assert '"tool"' in src and "TOOL RESULT" in src and "tool results" in JUDGE  # D14: true looked-up facts were judged invented
