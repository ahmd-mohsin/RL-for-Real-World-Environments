"""m13 C-7: context strategies (A2-46), compaction as a policy action (A2-47), thinking-history modes (A2-48)."""

from __future__ import annotations

import pytest

from crucible.actors.llm import LLMResponse
from crucible.actors.policy import SUMMARY_HEAD, ContextConfig, LLMToolPolicy
from crucible.core.schema.actions import Observation, ToolCall, ToolSpec


class Scripted:
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    async def complete(self, messages, **kw):
        self.seen.append([dict(m) for m in messages])
        r = self.replies.pop(0)
        return r if isinstance(r, LLMResponse) else LLMResponse(text=r, tool_calls=[], usage={"prompt_tokens": 1, "completion_tokens": 1}, model="m")


def _transcript(pol, turns=6, tool_chars=4000):
    pol.messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "THE TASK"}]
    for k in range(turns):
        pol.messages += [{"role": "assistant", "content": f"step {k}", "tool_calls": [{"id": f"c{k}", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                         {"role": "tool", "tool_call_id": f"c{k}", "content": f"out{k} " + "x" * tool_chars}]


@pytest.mark.parametrize("strategy,expect", [("keep_recent", "keep_recent"), ("discard_all", "discard_all"), ("summary", "summary"),
                                             ("tiered", "tiered:shrink"), ("hybrid", "hybrid:summary")])
async def test_each_strategy_on_a_scripted_transcript(strategy, expect):
    llm = Scripted(["SUMMARY TEXT"])
    budget = 6000 if strategy == "tiered" else 2000  # tiered: shrinking the 4 older outputs is enough at this budget
    pol = LLMToolPolicy("agent", llm, context=ContextConfig(budget_tokens=budget, trigger=0.5, keep_recent_turns=2, strategy=strategy, tier_tool_chars=50))
    _transcript(pol)
    await pol._maybe_compact()
    seg = pol.segments[-1]
    assert seg["strategy"] == expect and pol.segment == 1 and seg["context_after"] < seg["context_before"]
    texts = [str(m.get("content")) for m in pol.messages]
    assert texts[0] == "sys" and texts[1] == "THE TASK"  # system + task always kept verbatim
    if expect == "keep_recent":
        assert sum(m["role"] == "assistant" for m in pol.messages) == 2 and not llm.seen  # no model call
    if expect == "discard_all":
        assert not any(m["role"] == "assistant" for m in pol.messages) and "discarded" in texts[2] and not llm.seen
    if expect in ("summary", "hybrid:summary"):
        assert texts[2] == SUMMARY_HEAD + "SUMMARY TEXT" and pol.model_calls[-1]["kind"] == "compaction" and seg["summary_call"]
    if expect == "tiered:shrink":
        assert not llm.seen and any("elided" in t for t in texts) and sum(m["role"] == "assistant" for m in pol.messages) == 6


async def test_hybrid_discards_after_its_summaries_keeping_the_latest_summary():
    pol = LLMToolPolicy("agent", Scripted(["S1"]), context=ContextConfig(budget_tokens=2000, trigger=0.5, keep_recent_turns=1, strategy="hybrid"))
    _transcript(pol)
    await pol._maybe_compact()
    _transcript_more = [{"role": "assistant", "content": "more", "tool_calls": []}, {"role": "user", "content": "y" * 8000}]
    pol.messages += _transcript_more
    await pol._maybe_compact()
    assert [s["strategy"] for s in pol.segments] == ["hybrid:summary", "hybrid:discard_all"]
    assert pol.messages[2]["content"] == SUMMARY_HEAD + "S1" and len(pol.messages) == 3


async def test_tiered_falls_back_to_a_summary_when_shrinking_is_not_enough():
    pol = LLMToolPolicy("agent", Scripted(["S"]), context=ContextConfig(budget_tokens=400, trigger=0.5, keep_recent_turns=6, strategy="tiered"))
    _transcript(pol)  # every turn is "recent": nothing to shrink
    await pol._maybe_compact()
    assert pol.segments[-1]["strategy"] == "tiered:summary"


async def test_compaction_as_a_policy_action():
    compact = LLMResponse(text="", tool_calls=[{"id": "k1", "name": "context_compact", "arguments": {"summary": "done: A; next: B"}}],
                          usage={"prompt_tokens": 1, "completion_tokens": 1}, model="m")
    act = LLMResponse(text="", tool_calls=[{"id": "t1", "name": "ws__read", "arguments": {"path": "a"}}], usage={"prompt_tokens": 1, "completion_tokens": 1}, model="m")
    llm = Scripted([compact, act])
    pol = LLMToolPolicy("agent", llm, context=ContextConfig(budget_tokens=10, trigger=0.1, mode="action"))  # far over budget: no auto trigger
    obs = Observation(actor="agent", step=0, instruction="THE TASK", tools=[ToolSpec(app="ws", name="read", description="r", input_schema={"type": "object"})])
    a = await pol.act(obs)
    assert isinstance(a, ToolCall) and a.function_name == "ws.read"
    assert any(t["function"]["name"] == "context_compact" for t in pol.tools)
    assert [c["kind"] for c in pol.model_calls] == ["compaction", "turn"] and pol.segments[0]["strategy"] == "action"
    assert pol.segments[0]["summary_call"] == pol.model_calls[0]["call_id"] and pol.model_calls[1]["segment"] == 1
    assert SUMMARY_HEAD + "done: A; next: B" in [m["content"] for m in llm.seen[1]]  # the second call saw the summary, not the history


async def test_repeated_compaction_without_acting_stops():
    compact = LLMResponse(text="", tool_calls=[{"id": "k", "name": "context_compact", "arguments": {"summary": "s"}}], usage={}, model="m")
    pol = LLMToolPolicy("agent", Scripted([compact] * 4), context=ContextConfig(mode="action"))
    a = await pol.act(Observation(actor="agent", step=0, instruction="T", tools=[]))
    assert pol.stop == "too_many_compactions" and a.reason


def test_thinking_modes_follow_the_template():
    from transformers import AutoTokenizer

    from crucible.train.chat import ByteChatCodec, HFChatCodec
    from crucible.train.world_env import EngineLLM

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except OSError:
        pytest.skip("Qwen3 tokenizer not cached")
    qwen = HFChatCodec(tok, template_kwargs={"enable_thinking": False})

    async def gen(prompt, n, t):  # pragma: no cover - not called
        raise AssertionError

    assert EngineLLM(qwen, gen).thinking == "interleaved"  # Qwen3 keeps reasoning only after the last user message
    with pytest.raises(ValueError, match="training contexts would differ"):
        EngineLLM(qwen, gen, thinking="preserved")
    with pytest.raises(ValueError):
        EngineLLM(ByteChatCodec(), gen, thinking="strip")
    assert EngineLLM(ByteChatCodec(), gen).thinking == "preserved"


async def test_interleaved_extends_in_the_tool_loop_and_rerenders_on_a_user_turn():
    from transformers import AutoTokenizer

    from crucible.train.chat import HFChatCodec
    from crucible.train.engine import GenResult
    from crucible.train.world_env import EngineLLM

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except OSError:
        pytest.skip("Qwen3 tokenizer not cached")
    codec = HFChatCodec(tok, template_kwargs={"enable_thinking": False})
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    reply = tok.encode("<think>plan</think>calling\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>", add_special_tokens=False) + [eos]

    async def gen(prompt, n, t):
        return GenResult(reply, [-0.1] * len(reply), 0, "stop")

    llm = EngineLLM(codec, gen)
    m = [{"role": "user", "content": "go"}]
    r1 = await llm.complete(m)
    full1 = r1.tokens["prompt_token_ids"] + r1.tokens["completion_token_ids"]
    m += [{"role": "assistant", "content": r1.text, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
          {"role": "tool", "tool_call_id": "a", "content": "ok"}]
    r2 = await llm.complete(m)
    assert r2.tokens["prompt_token_ids"][: len(full1)] == full1 and "new_segment" not in r2.tokens  # inside the tool loop: extends
    m += [{"role": "assistant", "content": r2.text}, {"role": "user", "content": "now something else"}]
    r3 = await llm.complete(m)
    assert r3.tokens.get("new_segment") and r3.tokens["prompt_token_ids"] == codec.render(m, None) and llm.rerenders == 1
