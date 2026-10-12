"""M2 audit (2026-10-12), batch 1: the policy loop and the LLM client. Each test reproduced a defect before its fix
(``docs/audit/2026-10-12/M02-actors.md`` §2, rows L1–L14)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from crucible.actors.llm import ContextOverflow, LLMResponse
from crucible.actors.policy import ContextConfig, EpisodeLimits, LLMToolPolicy, run_episode
from crucible.core.schema.actions import Done, Observation, ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import World


class Script:
    """A fake LLM: scripted replies (str, LLMResponse or an exception to raise), prompts recorded."""

    def __init__(self, replies: list):
        self.replies, self.prompts, self.model, self.keys = list(replies), [], "fake", []

    async def complete(self, messages, *, tools=None, **kw):
        self.prompts.append(messages)
        self.keys.append(kw.get("sample_key"))
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r if isinstance(r, LLMResponse) else LLMResponse(text=r)


def call(name: str, args: dict, i: int = 0, **kw) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[{"id": f"c{i}", "name": name, "arguments": args}], **kw)


def obs(**kw) -> Observation:
    return Observation(actor="agent", step=0, **kw)


async def _world(tmp_path: Path) -> World:
    spec = WorldSpec.model_validate({"world": "t/loop", "apps": [{"id": "erp", "template": "common/json_api", "state": {"adapter": "json_doc", "params": {"path": "erp.json"}},
                                                                   "params": {"collections": ["orders"]}}],
                                     "actors": [{"id": "agent", "kind": "policy", "tools": ["erp.*"]}]})
    w = World(spec, instructions={"agent": "Do the task."}, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    return w


async def test_L1_the_callers_sample_key_survives_run_episode(tmp_path):
    w = await _world(tmp_path)
    llm = Script(["done"])
    p = LLMToolPolicy("agent", llm)
    try:
        await run_episode(w, {"agent": p}, sample_key="attempt-7")
    finally:
        await w.close()
    assert llm.keys == ["attempt-7"]  # was overwritten with the World's random episode id: replay never hit, every replay masked


async def test_L2_a_failed_compaction_call_ends_the_turn_instead_of_crashing():
    big = "x" * 4000
    p = LLMToolPolicy("agent", Script([ContextOverflow("prompt too long")]), context=ContextConfig(budget_tokens=500, strategy="summary"))
    p.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": big}]
    a = await p.act(obs())
    assert isinstance(a, Done) and p.stop == "context_overflow"


async def test_L3_an_empty_summary_never_wipes_the_history():
    big = "fact-" * 1000
    p = LLMToolPolicy("agent", Script([LLMResponse(text="", raw_finish="length"), LLMResponse(text="", raw_finish="length"), "done"]),
                      context=ContextConfig(budget_tokens=500, strategy="summary", keep_recent_turns=1))
    p.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "the task"}, {"role": "user", "content": big}]
    await p.act(obs())
    assert p.segments and "summary_failed" in p.segments[-1]["strategy"]  # fell back to keeping recent turns, recorded
    assert not any(m.get("content") == "" and "summary" in str(m).lower() for m in p.messages)  # no empty summary in the context


async def test_L5_policy_stop_labels_reach_the_episode_result(tmp_path):
    w = await _world(tmp_path)
    comp = {"summary": "s"}
    llm = Script([call("context_compact", comp, 1), call("context_compact", comp, 2), call("context_compact", comp, 3)])
    try:
        res = await run_episode(w, {"agent": LLMToolPolicy("agent", llm, context=ContextConfig(mode="action"))})
    finally:
        await w.close()
    assert res.stop_condition == "too_many_compactions" and res.truncated  # was reported as a voluntary agent_done


async def test_L6_L7_nudges_reset_per_clean_step_and_keep_the_game_context():
    blank = LLMResponse(text="")
    p = LLMToolPolicy("agent", Script([blank, call("erp__get", {"id": 1}, 1), blank, blank, call("erp__get", {"id": 2}, 2)]))
    p._names = {"erp__get": "erp.get"}
    a1 = await p.act(obs())
    p._pending.clear()
    a2 = await p.act(obs())  # two more blank replies later in the episode: still nudged, not an agent_done
    assert isinstance(a1, ToolCall) and isinstance(a2, ToolCall)
    seen: list = []

    class Spy(Script):
        async def complete(self, messages, **kw):
            seen.append(messages[-1]["content"])
            return await super().complete(messages, **kw)

    q = LLMToolPolicy("agent", Spy([blank, "I'll wait", call("nim__take", {"n": 1}, 3)]))
    q._names = {"nim__take": "nim.take"}
    legal = [{"function_name": "nim.take", "arguments": {"n": 1}}]
    a = await q.act(obs(legal=legal, to_move=["agent"]))
    assert isinstance(a, ToolCall)  # the empty reply's nudge dropped `legal`, so the text reply resigned the seat


def _fake_openai(responses: list):
    class Fake:
        chat = property(lambda self: self)
        completions = property(lambda self: self)

        async def create(self, **kw):
            r = responses.pop(0)
            if isinstance(r, BaseException):
                raise r
            return r

    return Fake()


def _resp(text: str = "ok", *, cost=None, finish="stop", tool_calls=None):
    msg = NS(content=text, tool_calls=tool_calls, model_dump=lambda exclude_none=True: {"role": "assistant", "content": text})
    usage = NS(prompt_tokens=1000, completion_tokens=500, cost=cost, completion_tokens_details=None, prompt_tokens_details=None, model_extra={})
    return NS(choices=[NS(message=msg, finish_reason=finish, logprobs=None)], usage=usage, model="fake", id="x")


async def test_L9_an_overflow_in_a_200_error_body_is_an_overflow(tmp_path):
    from crucible.actors.llm import LLMClient

    c = LLMClient("policy-default", budget_usd=1, cache_mode="off", cache_dir=tmp_path)
    c._client = _fake_openai([NS(choices=[], error={"message": "This model's maximum context length is 32768 tokens"})] * 3)
    with pytest.raises(ContextOverflow):  # was retried 3 times and masked as provider_unavailable
        await c.complete([{"role": "user", "content": "q"}])


async def test_L10_an_unpriced_model_still_spends_against_the_budget(tmp_path):
    from crucible.actors.llm import BudgetExceeded, LLMClient

    c = LLMClient("openrouter:vendor/unpriced-model", budget_usd=0.05, cache_mode="off", cache_dir=tmp_path)
    c._client = _fake_openai([_resp(cost=None) for _ in range(200)])
    with pytest.raises(BudgetExceeded):  # every call cost $0: the hard budget could never trip
        for _ in range(200):
            await c.complete([{"role": "user", "content": "q"}])
    assert c.spend.unpriced_calls > 0 and c.spend.uncertain_usd > 0


async def test_L13_a_masking_simulator_failure_stops_the_episode(tmp_path):
    w = await _world(tmp_path)

    class Breaks:
        actor_id = "cust"

        def __init__(self):
            self.failures: list = []

        async def respond(self, turn):
            return []

        def state_dict(self):
            return {}

        def load_state_dict(self, d):
            pass

    br = Breaks()
    w.responders["cust"] = br
    llm = Script([call("erp__list", {}, i) for i in range(30)])
    p = LLMToolPolicy("agent", llm)
    p._names = {"erp__list": "erp.list"}
    orig = w.step

    async def step(actor, action):
        r = await orig(actor, action)
        br.failures.append({"kind": "crucible:sim_failure", "reason": "broke character"})  # fails after the first step
        return r

    w.step = step
    try:
        res = await run_episode(w, {"agent": p}, limits=EpisodeLimits(max_steps=30))
    finally:
        await w.close()
    assert res.masked and res.steps <= 2 and res.stop_condition == "crucible:sim_failure"  # ran (and paid) to max_steps


async def test_L14_a_corrupt_cache_entry_is_a_miss(tmp_path):
    from crucible.actors.llm import LLMClient

    c = LLMClient("policy-default", budget_usd=1, cache_mode="record", cache_dir=tmp_path)
    c._client = _fake_openai([_resp("first", cost=0.0001), _resp("again", cost=0.0001)])
    await c.complete([{"role": "user", "content": "q"}])
    [f] = list(tmp_path.rglob("*.json"))
    f.write_text('{"text": "half wri')  # a crash mid-write (before atomic writes)
    r = await c.complete([{"role": "user", "content": "q"}])
    assert r.text == "again"  # raised JSONDecodeError on every later run


async def test_L15_a_cut_off_reply_does_not_execute_its_tool_calls():
    cut = call("erp__create", {"collection": "orders"}, 1, raw_finish="length")
    p = LLMToolPolicy("agent", Script([cut, call("erp__get", {"id": 1}, 2)]))
    p._names = {"erp__create": "erp.create", "erp__get": "erp.get"}
    a = await p.act(obs())
    assert isinstance(a, ToolCall) and a.function_name == "erp.get"  # the truncated batch used to run
    assert any("cut off" in str(m.get("content", "")) for m in p.messages)


async def test_L4_compaction_triggers_on_the_providers_token_count():
    dense = "数据" * 300  # ~600 tokens to a real tokenizer, ~150 by chars/4
    p = LLMToolPolicy("agent", Script([LLMResponse(text="", tool_calls=[{"id": "c1", "name": "erp__get", "arguments": {}}],
                                                   usage={"prompt_tokens": 900, "completion_tokens": 10}), "summary of the work", "done"]),
                      context=ContextConfig(budget_tokens=1000, strategy="summary"))
    p._names = {"erp__get": "erp.get"}
    p.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": dense}]
    assert p.context_tokens() < 800  # the estimate alone never reaches the trigger
    await p.act(obs())
    assert p.context_tokens() >= 900  # measured: the provider counted 900 prompt tokens
    from crucible.core.schema.actions import ToolResult

    await p.act(obs(tool_results=[ToolResult(tool_call_id="c1", function_name="erp.get", ok=True, content="{}")]))
    assert p.segments and p.segments[0]["strategy"] == "summary"  # compacted before the provider refused the prompt


async def test_L8_L12_timeouts_and_rate_limits(tmp_path, monkeypatch):
    from crucible.actors import llm as m

    assert m.LLMClient("policy-default", cache_dir=tmp_path).timeout_s == 600  # was a fixed 120 s: reasoning models timed out after paying
    monkeypatch.setenv("CRUCIBLE_LLM_TIMEOUT_S", "45")
    assert m.LLMClient("policy-default", cache_dir=tmp_path).timeout_s == 45

    class RateLimited(Exception):
        status_code = 429

    sleeps: list[float] = []

    async def no_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(m.asyncio, "sleep", no_sleep)
    c = m.LLMClient("policy-default", budget_usd=1, cache_mode="off", cache_dir=tmp_path)
    c._client = _fake_openai([RateLimited("slow down")] * 4 + [_resp("ok", cost=0.0001)])
    r = await c.complete([{"role": "user", "content": "q"}])
    assert r.text == "ok" and len(sleeps) == 4 and min(sleeps) >= 2.5  # 3 attempts over ~4.5 s gave up mid-burst


async def test_L16_a_content_filter_is_a_labelled_stop_not_an_empty_reply():
    p = LLMToolPolicy("agent", Script([LLMResponse(text="", raw_finish="content_filter")]))
    a = await p.act(obs())
    assert isinstance(a, Done) and p.stop == "content_filter" and len(p.llm.prompts) == 1  # was nudged as blank, twice
