"""M2 R-6 (audit 2026-10-15): every tool call of an episode has a unique, provider-safe id that is the same on a replay.
Ids passed unchecked and empty ones got a random id, though they key the World's results, the batch in flight and the
format penalties (probes v1_ng6, ng01, v2_mv2)."""

from __future__ import annotations

import json
from pathlib import Path

from crucible.actors.harness import BashLoopPolicy
from crucible.actors.llm import LLMResponse
from crucible.actors.policy import CALL_ID, LLMToolPolicy, run_episode
from crucible.core.schema.actions import Inbound, Observation, ToolCall, ToolResult
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.credit.shaping import format_errors_by_call
from crucible.world.world import World

SPEC = {"world": "probe/id-reuse", "actors": [{"id": "agent", "kind": "policy", "tools": ["erp.*"]}],
        "apps": [{"id": "erp", "template": "common/json_api", "state": {"adapter": "json_doc", "params": {"path": "erp.json"}},
                  "params": {"collections": ["orders"]}}]}
CREATE = "erp__create_resource"


class Fake:
    def __init__(self, replies):
        self.replies, self.model, self.prompts = list(replies), "fake", []

    async def complete(self, messages, *, tools=None, **kw):
        self.prompts.append(json.loads(json.dumps(messages, default=str)))
        return self.replies.pop(0) if self.replies else LLMResponse(text="All done.")


def order(i: str, ident: str | None = "call_0") -> dict:
    return {"id": ident, "name": CREATE, "arguments": {"collection": "orders", "id": i, "data": {"qty": 1}}}


async def _episode(tmp: Path, replies) -> tuple:
    w = World(WorldSpec.model_validate(SPEC), store=Store(tmp / "s"), workdir=tmp / "w", instructions={"agent": "create orders"})
    await w.start()
    try:
        res = await run_episode(w, {"agent": LLMToolPolicy("agent", Fake(replies))}, max_steps=10)
    finally:
        await w.close()
    return res, [e.data.get("id") for e in res.trace.events if e.type == "tool_call"]


async def test_an_id_reused_across_turns_charges_only_its_own_call(tmp_path):
    bad = {"id": "call_0", "name": CREATE, "arguments": {"__unparseable__": '{"collection": "ord'}}
    res, ids = await _episode(tmp_path, [LLMResponse(text="", tool_calls=[order("O1")]), LLMResponse(text="", tool_calls=[bad])])
    assert ids[0] == "call_0" and ids[1] != "call_0" and CALL_ID.fullmatch(ids[1])  # the first keeps its id, the repeat is replaced
    assert format_errors_by_call(res.trace, res.model_calls) == {"agent:c2": 1}  # was {'agent:c1': 1, 'agent:c2': 1}
    assert res.model_calls["agent"][1]["provider_ids"] == ["call_0"]


async def test_two_calls_sharing_an_id_keep_their_batch_together():
    llm = Fake([LLMResponse(text="", tool_calls=[order("O1", "c1"), order("O2", "c1")])])
    p = LLMToolPolicy("agent", llm)
    a1 = await p.act(Observation(actor="agent", step=0, instruction="create two orders"))
    a2 = await p.act(Observation(actor="agent", step=1, inbox=[Inbound(**{"from": "boss", "text": "only one", "at": None})],
                                 tool_results=[ToolResult(tool_call_id=a1.tool_call_id, function_name="erp.create_resource", ok=True, content="ok O1")]))
    assert isinstance(a2, ToolCall) and a2.tool_call_id != a1.tool_call_id
    await p.act(Observation(actor="agent", step=2, tool_results=[ToolResult(tool_call_id=a2.tool_call_id, function_name="erp.create_resource",
                                                                            ok=True, content="ok O2")]))
    roles = [(m["role"], m.get("tool_call_id")) for m in llm.prompts[-1][2:]]
    assert roles == [("assistant", None), ("tool", "c1"), ("tool", a2.tool_call_id), ("user", None)]  # was: the user message between


async def test_id_less_and_unsafe_ids_are_derived_the_same_on_every_replay(tmp_path):
    replies = [LLMResponse(text="", tool_calls=[order("O1", None), order("O2", "functions.create:1")])]
    _, first = await _episode(tmp_path / "a", list(replies))
    _, again = await _episode(tmp_path / "b", list(replies))
    assert first == again and all(i and CALL_ID.fullmatch(i) and len(i) == 9 for i in first)  # was a random uuid4, and ':' kept


def test_a_restored_policy_still_knows_the_ids_it_used():
    p = LLMToolPolicy("agent", Fake([]))
    p.messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": CREATE, "arguments": "{}"}}]})
    q = LLMToolPolicy("agent", Fake([]))
    q.load_state_dict(p.state_dict())
    assert "call_0" in q._ids_seen


async def test_the_bash_loop_numbers_its_calls():
    llm = Fake([LLMResponse(text="```bash\nls\n```"), LLMResponse(text="```bash\npwd\n```")])
    p = BashLoopPolicy("agent", llm)
    a = await p.act(Observation(actor="agent", step=0, instruction="look around"))
    b = await p.act(Observation(actor="agent", step=1, tool_results=[ToolResult(tool_call_id=a.tool_call_id, function_name="ws.exec", ok=True, content="x")]))
    assert (a.tool_call_id, b.tool_call_id) == ("bash-1", "bash-2")  # was bash-<random>
