"""M2 LLM-backed actors with a fake LLM (no network, no cost): quote mode, paraphrase checking,
scenario simulator regeneration/masking, and the tool-calling policy loop."""

from __future__ import annotations

import json
from pathlib import Path

from crucible.actors.grounded import GroundedResponder
from crucible.actors.grounding import NO_RECORD
from crucible.actors.llm import LLMResponse
from crucible.actors.policy import LLMToolPolicy, run_episode
from crucible.actors.records import RecordStore
from crucible.actors.simulated import DONE, Scenario, SimulatedActor
from crucible.core.schema.actions import Message
from crucible.core.schema.trace import Record
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import ActorTurn, World


class FakeLLM:
    """Returns scripted responses in order; records prompts."""

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.prompts: list = []
        self.model = "fake"

    async def complete(self, messages, *, tools=None, **kw):
        self.prompts.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, LLMResponse):
            return r
        return LLMResponse(text=r)

    async def complete_json(self, messages, **kw):
        r = await self.complete(messages, **kw)
        return json.loads(r.text)


RECS = RecordStore([
    Record(id="a", source="s", role="reviewer", time="2026-01-01T00:00:00Z", text="As far as I see, the stride is never checked in fps_cuda.cu."),
    Record(id="b", source="s", role="reviewer", time="2026-01-02T00:00:00Z", text="Please add a test for empty point clouds in fps_cuda.cu."),
])


async def test_quote_mode_returns_records_verbatim():
    llm = FakeLLM(['{"notes": [1], "lead": "On that file I wrote:"}', '{"notes": []}', '{"notes": [1], "lead": "The stride was checked 3 times"}'])
    r = GroundedResponder("rev", RECS, mode="quote", llm=llm, min_score=0.1)
    text, cites = await r.answer("what about the stride in fps_cuda.cu?", now="2026-02-01T00:00:00Z")
    assert text == "As far as I see, the stride is never checked in fps_cuda.cu." and cites == ["a"]  # pure quote
    assert (await r.answer("stride fps_cuda?", now="2026-02-01T00:00:00Z"))[0] == NO_RECORD
    text, _ = await r.answer("stride fps_cuda?", now="2026-02-01T00:00:00Z")
    assert "3 times" not in text  # whatever the model adds is never shown; only the record speaks


async def test_paraphrase_mode_drops_unsupported_after_regeneration():
    llm = FakeLLM([
        "The stride is never checked in fps_cuda.cu [R:a]. It also leaks 512 MB per call [R:a].",
        "The stride is never checked in fps_cuda.cu [R:a]. It also leaks 512 MB per call [R:a].",
    ])
    r = GroundedResponder("rev", RECS, mode="llm", llm=llm, min_score=0.1)
    text, cites = await r.answer("stride fps_cuda?", now="2026-02-01T00:00:00Z")
    assert "512" not in text and "never checked" in text and cites == ["a"]
    assert r.stats["regenerations"] == 1 and r.stats["dropped_sentences"] == 1


async def test_scenario_sim_regenerates_on_judge_reject_and_masks_persistent_failure():
    sc = Scenario(persona="Lee", reason_for_contact="double charge of 39.50 USD")
    sim = SimulatedActor("lee", sc, FakeLLM(["I was charged 99 USD.", "I was charged twice, 39.50 USD each."]),
                         judge=FakeLLM(['{"consistent": false, "reason": "99 not in scenario"}', '{"consistent": true, "reason": ""}']))
    out = await sim.respond(ActorTurn(actor="lee", time=None, inbox=[_inbound("agent", "How much were you charged?")]))
    assert isinstance(out[0], Message) and "39.50" in out[0].text and not sim.failed
    assert [a["consistent"] for a in sim.audit] == [False, True]
    bad = SimulatedActor("lee", sc, FakeLLM(["x1", "x2", "x3"]), judge=FakeLLM(['{"consistent": false}'] * 3), max_regenerations=2)
    assert await bad.respond(ActorTurn(actor="lee", time=None, inbox=[_inbound("agent", "hi")])) == []
    assert bad.failed and bad.failures[0]["kind"] == "crucible:sim_failure"
    done = SimulatedActor("lee", sc, FakeLLM([DONE]))
    assert await done.respond(ActorTurn(actor="lee", time=None, inbox=[_inbound("agent", "refunded")])) == [] and done.finished


def _inbound(frm: str, text: str):
    from crucible.core.schema.actions import Inbound

    return Inbound(**{"from": frm, "text": text})


async def test_llm_tool_policy_episode(tmp_path: Path):
    spec = WorldSpec.model_validate({
        "world": "test/policy",
        "apps": [{"id": "erp", "template": "common/json_api", "state": {"adapter": "json_doc", "params": {"path": "erp.json"}}, "params": {"collections": ["orders"]}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["erp.*"]}, {"id": "boss", "kind": "simulated"}],
    })
    llm = FakeLLM([
        LLMResponse(text="", tool_calls=[{"id": "c1", "name": "erp__create_resource", "arguments": {"collection": "orders", "id": "O1", "data": {"qty": 2}}}]),
        LLMResponse(text="", tool_calls=[{"id": "c2", "name": "send_message", "arguments": {"to": "boss", "text": "created O1"}}]),
        LLMResponse(text="All done."),
    ])
    w = World(spec, store=Store(tmp_path / "s"), workdir=tmp_path / "w", instructions={"agent": "create order O1"})
    w.attach(GroundedResponder("boss", RecordStore([]), mode="verbatim"))
    await w.start()
    try:
        res = await run_episode(w, {"agent": LLMToolPolicy("agent", llm)}, max_steps=10)
        assert res.done and res.steps == 3
        assert (await w.verifier_view().query("erp", {"op": "get", "args": {"path": "/orders/O1/qty"}})).value == 2
        # the policy saw the tool result and the boss's reply on later turns
        flat = json.dumps(llm.prompts[-1])
        assert "created" in flat and NO_RECORD in flat
    finally:
        await w.close()
