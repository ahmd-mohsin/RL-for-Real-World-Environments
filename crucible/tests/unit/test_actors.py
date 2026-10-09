"""M2 actors without paid LLM calls: records/time, grounding checks, grounded/relay/CI/human/peer actors
inside real Worlds, and actor memory in snapshots."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from crucible.actors.grounded import GroundedResponder
from crucible.actors.grounding import NO_RECORD, check, is_factual
from crucible.actors.human import HumanActor, PeerActor
from crucible.actors.policy import ScriptedPolicy, run_episode
from crucible.actors.records import RecordStore
from crucible.actors.scripted import Check, CheckRunnerActor, RelayActor, Rule
from crucible.actors.transports import CallableTransport, FileQueueTransport
from crucible.core.schema.actions import Message, ToolCall
from crucible.core.schema.trace import Record
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import World

RECORDS = [
    Record(id="pr1#3", source="pr1", author="r1", role="reviewer", time="2026-01-10T10:00:00Z",
           text="tex1Dfetch returned 0 out of range; __ldg reads garbage past the end of the buffer in multi_tree.cu line 88."),
    Record(id="pr1#5", source="pr1", author="r1", role="reviewer", time="2026-01-11T09:00:00Z",
           text="Please keep the guard band at 1 KiB alignment; 256 B breaks the original status codes."),
    Record(id="pr2#2", source="pr2", author="r2", role="reviewer", time="2026-06-01T00:00:00Z",
           text="Approved after the bounds check landed; latency is 12.5 ms on the 5090."),
    Record(id="fin#1", source="fin", author="cfo", role="approver", time="2026-01-01T00:00:00Z",
           text="经理审批 refunds above 50 USD require supervisor approval per policy FIN-7."),
]


def store() -> RecordStore:
    return RecordStore(RECORDS)


def test_records_time_visibility_and_bm25():
    s = store()
    hits = s.search("out of range texture fetch garbage", k=2)
    assert hits[0][0].id == "pr1#3"
    assert s.search("bounds check latency", now="2026-02-01T00:00:00Z") == [] or all(r.id != "pr2#2" for r, _ in s.search("bounds check latency", now="2026-02-01T00:00:00Z"))
    assert s.get("pr2#2", now="2026-02-01T00:00:00Z") is None and s.get("pr2#2") is not None
    assert s.search("审批 refunds", k=1)[0][0].id == "fin#1"  # CJK tokens retrieve
    # instants, not strings: a record at ...10:00:00Z is visible at ...10:00:00.5Z
    assert s.get("pr1#3", now="2026-01-10T10:00:00.500000Z") is not None and s.get("pr1#3", now="2026-01-10T09:59:59.9Z") is None


def test_grounding_checker():
    s = store()
    ok = check("The fetch returned 0 out of range in multi_tree.cu line 88 [R:pr1#3].", s, now="2026-03-01T00:00:00Z")
    assert ok.ok and ok.citations == ["pr1#3"]
    nocite = check("The kernel reads garbage past the end of the buffer.", s)
    assert not nocite.ok and nocite.unsupported[0].reason == "no citation"
    future = check("Latency is 12.5 ms on the 5090 [R:pr2#2].", s, now="2026-02-01T00:00:00Z")
    assert not future.ok and "future" in future.unsupported[0].reason
    wrong_number = check("Keep the guard band at 4 KiB alignment [R:pr1#5].", s)
    assert not wrong_number.ok and "facts not in cited records" in wrong_number.unsupported[0].reason
    unrelated = check("The deployment pipeline uses blue green rollouts everywhere [R:pr1#5].", s)
    assert not unrelated.ok and "overlap" in unrelated.unsupported[0].reason
    assert not is_factual("Thanks!") and not is_factual("Could you share the line number?") and not is_factual(NO_RECORD)
    mixed = check("Thanks. tex1Dfetch returned 0 out of range [R:pr1#3]. Also the code has 99 bugs.", s)
    assert len(mixed.unsupported) == 1 and "99" not in mixed.clean_text() and "tex1Dfetch" in mixed.clean_text()


def world_spec() -> WorldSpec:
    return WorldSpec.model_validate({
        "world": "test/actors",
        "apps": [{"id": "erp", "template": "common/json_api", "state": {"adapter": "json_doc", "params": {"path": "erp.json"}},
                  "params": {"collections": ["orders"]}},
                 {"id": "box", "template": "common/toolchain", "state": {"adapter": "process", "params": {"cwd": "."}}}],
        "actors": [
            {"id": "agent", "kind": "policy", "tools": ["erp.*"]},
            {"id": "reviewer", "kind": "simulated"},
            {"id": "clerkbot", "kind": "system", "tools": ["erp.get_resource"]},
            {"id": "ci", "kind": "system"},
            {"id": "boss", "kind": "human"},
            {"id": "partner", "kind": "peer"},
        ],
        "clock": {"mode": "step", "t0": "2026-03-01T09:00:00Z", "step_dt_s": 60},
    })


async def test_actors_in_world_and_snapshot(tmp_path: Path):
    w = World(world_spec(), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    rev = GroundedResponder("reviewer", store(), who="a senior reviewer", mode="verbatim", roles={"reviewer"}, reply_latency="30m")
    bot = RelayActor("clerkbot", [Rule(r"status of (\w+)", tool="erp.get_resource", args={"collection": "orders", "id": "{0}"}, reply="Order status: {tool_result}"),
                                  Rule(r"hello", reply="Hi, I am the clerk bot.")], default="Sorry, I only answer order questions.")
    answers = []

    async def boss_says(text: str) -> str:
        answers.append(text)
        return "Approved. Go ahead."

    w.attach(rev)
    w.attach(bot)
    w.attach(CheckRunnerActor("ci", w.verifier_view(), [Check("unit", "box", "true"), Check("lint", "box", "false")]))
    w.attach(HumanActor("boss", CallableTransport(boss_says)))
    await w.start()
    try:
        await w.reset()
        await w.step("agent", ToolCall(tool_call_id="1", function_name="erp.create_resource", arguments={"collection": "orders", "id": "O7", "data": {"status": "shipped"}}))
        r = await w.step("agent", Message(to="reviewer", text="Any concerns about the out of range texture fetch?"))
        assert "garbage past the end" in r.observation.inbox[0].text and r.observation.inbox[0].citations == ["pr1#3"]
        assert w.clock.now_iso() >= "2026-03-01T09:30:00Z"  # reviewer reply latency advanced virtual time
        r = await w.step("agent", Message(to="reviewer", text="What is our policy on cafeteria menus?"))
        assert r.observation.inbox[0].text == NO_RECORD
        r = await w.step("agent", Message(to="clerkbot", text="what is the status of O7"))
        assert "shipped" in r.observation.inbox[0].text
        r = await w.step("agent", Message(to="clerkbot", text="tell me a joke"))
        assert "only answer order" in r.observation.inbox[0].text
        r = await w.step("agent", Message(to="ci", text="please run the checks"))
        assert "unit: PASS" in r.observation.inbox[0].text and "lint: FAIL" in r.observation.inbox[0].text
        ref = await w.snapshot()
        r = await w.step("agent", Message(to="boss", text="May I refund 80 USD?"))
        assert r.observation.inbox[0].text == "Approved. Go ahead." and "[from agent] May I refund 80 USD?" in answers[0]
        assert len(rev.history) == 2 and ref.actors["reviewer"]["stats"]["replies"] == 2
        await w.restore(ref)
        assert len(w.responders["boss"].history) == 0 and w.responders["ci"].runs == 1
    finally:
        await w.close()


async def test_file_queue_human(tmp_path: Path):
    q = tmp_path / "q"
    human = HumanActor("boss", FileQueueTransport(q, timeout_s=10, poll_s=0.05))

    async def answer_later():
        for _ in range(100):
            await asyncio.sleep(0.05)
            if (q / "q-0001.json").exists():
                (q / "a-0001.json").write_text(json.dumps({"text": "Use vendor B."}))
                return

    w = World(world_spec(), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    w.attach(human)
    await w.start()
    try:
        await w.reset()
        task = asyncio.create_task(answer_later())
        r = await w.step("agent", Message(to="boss", text="Which vendor?"))
        await task
        assert r.observation.inbox[0].text == "Use vendor B."
    finally:
        await w.close()


try:  # module level: agentenv resolves run()'s TaskRequest annotation from module globals
    from agentenv_protocol.a2a_agent import AgentEnvAgent, AgentIdentity, TaskRequest, TaskResult, TextPart, a2a_agent

    @a2a_agent(identity=AgentIdentity(name="partner-agent", description="echo peer", version="1"))
    class Partner(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            text = "".join(p.text for p in request.parts if isinstance(p, TextPart))
            return TaskResult.builder().succeeded().add_text(f"partner received: {text}").build()
except ImportError:  # pragma: no cover
    Partner = None


async def test_peer_over_agentenv_a2a(tmp_path: Path):
    if Partner is None:
        pytest.skip("agentenv not installed")
    from crucible.interop.agentenv.a2a import A2ATransport, serve_agent

    served = await serve_agent(Partner())
    w = World(world_spec(), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    peer = PeerActor("partner", A2ATransport(served.url, timeout_s=30, poll_s=1))
    w.attach(peer)
    await w.start()
    try:
        await w.reset()
        r = await w.step("agent", Message(to="partner", text="Can you confirm the PO?"))
        assert r.observation.inbox[0].text == "partner received: [from agent] Can you confirm the PO?"
        st = (await w.snapshot()).actors["partner"]["transport"]
        assert st.get("context_id")  # the A2A conversation context is part of world state
    finally:
        await w.close()
        await served.stop()


async def test_scripted_policy_episode(tmp_path: Path):
    w = World(world_spec(), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    w.attach(GroundedResponder("reviewer", store(), mode="verbatim"))
    await w.start()
    try:
        pol = ScriptedPolicy("agent", [ToolCall(tool_call_id="1", function_name="erp.create_resource", arguments={"collection": "orders", "id": "A", "data": {}}),
                                       Message(to="reviewer", text="guard band alignment?")])
        res = await run_episode(w, {"agent": pol}, max_steps=10)
        assert res.done and res.steps == 3 and not res.masked
        assert any(e.actor == "reviewer" and "1 KiB" in (e.text or "") for e in res.trace.events)
    finally:
        await w.close()
