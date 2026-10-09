"""Toy pack end to end (CPU, no paid calls): the helpdesk world runs through M1 + M2 with an oracle
policy, and the verifier view sees exactly the state an expert would leave behind. Also served over
the agentenv backend to show the same world runs on a different framework unchanged."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.actors.policy import ScriptedPolicy, run_episode
from crucible.core.schema.actions import Message, ToolCall
from crucible.core.store import Store
from crucible.world.world import World
from crucible_pack_toy import INSTRUCTION, PACK_ROOT, helpdesk_spec, responders


def oracle() -> list:
    T = lambda i, f, a: ToolCall(tool_call_id=f"o{i}", function_name=f, arguments=a)  # noqa: E731
    return [
        T(1, "tickets.list_tickets", {}),
        T(2, "kb.read_doc", {"path": "refund_policy.md"}),
        # ticket 1: damaged blender (129.99 > 50 -> supervisor approval)
        Message(to="customer_dana", text="Hi Dana, what is wrong with the blender from order O-5001?"),
        Message(to="supervisor", text="Please approve a refund of 129.99 USD for ticket 1, order O-5001, damaged on arrival."),
        T(3, "tickets.issue_refund", {"ticket_id": 1, "order_id": "O-5001", "amount_usd": 129.99, "approved_by": "Sam Lee"}),
        T(4, "tickets.add_note", {"ticket_id": 1, "body": "Damaged on arrival (cracked jar). Full refund 129.99 USD approved by Sam Lee."}),
        T(5, "tickets.set_status", {"ticket_id": 1, "status": "resolved"}),
        # ticket 2: duplicate charge 39.50 (<= 50, no approval)
        Message(to="customer_lee", text="Lee, can you confirm the duplicate charge on order O-5002?"),
        T(6, "tickets.issue_refund", {"ticket_id": 2, "order_id": "O-5002", "amount_usd": 39.50}),
        T(7, "tickets.add_note", {"ticket_id": 2, "body": "Duplicate charge of 39.50 USD confirmed; refunded."}),
        T(8, "tickets.set_status", {"ticket_id": 2, "status": "resolved"}),
        # ticket 3: in transit -> not refundable, escalate
        T(9, "tickets.add_note", {"ticket_id": 3, "body": "Order O-5003 in transit; not refundable per FIN-7. Escalated for carrier trace."}),
        T(10, "tickets.set_status", {"ticket_id": 3, "status": "escalated"}),
        T(11, "mail.send_email", {"to": ["lee@mail.example"], "subject": "Your TV order O-5003", "body": "We opened a carrier trace and will update you."}),
    ]


async def _check_expert_end_state(w: World) -> None:
    v = w.verifier_view()
    rows = (await v.query("tickets", {"op": "sql", "args": {"sql": "SELECT id, status FROM tickets ORDER BY id"}})).value["rows"]
    assert rows == [[1, "resolved"], [2, "resolved"], [3, "escalated"]]
    refunds = (await v.query("tickets", {"op": "sql", "args": {"sql": "SELECT ticket_id, order_id, approved_by FROM refunds ORDER BY ticket_id"}})).value["rows"]
    assert refunds == [[1, "O-5001", "Sam Lee"], [2, "O-5002", None]]
    assert (await v.query("tickets", {"op": "count", "args": {"table": "notes"}})).value == 3


async def test_toy_helpdesk_oracle_episode(tmp_path: Path):
    w = World(helpdesk_spec(), instructions={"agent": INSTRUCTION}, store=Store(tmp_path / "s"), workdir=tmp_path / "w", pack_root=PACK_ROOT)
    for r in responders():
        w.attach(r)
    await w.start()
    try:
        res = await run_episode(w, {"agent": ScriptedPolicy("agent", oracle())}, max_steps=40)
        assert res.done and not res.masked
        ev = res.trace.events
        assert any(e.actor == "customer_dana" and "crack along the base" in (e.text or "") for e in ev)
        assert any(e.actor == "supervisor" and "Approved refund of 129.99 USD" in (e.text or "") for e in ev)
        assert any(e.type == "event:nl" for e in ev)  # SLA reminder fired (virtual time passed 1h)
        await _check_expert_end_state(w)
        # the repeating on_tool event sent one audit email per refund
        n_audit = (await w.verifier_view().query("mail", {"op": "scalar", "args": {"sql": "SELECT COUNT(*) FROM messages WHERE subject = 'refund logged'"}})).value
        assert n_audit == 2
    finally:
        await w.close()


async def test_toy_helpdesk_on_agentenv_gateway(tmp_path: Path):
    pytest.importorskip("agent_env")
    w = World(helpdesk_spec(), instructions={"agent": INSTRUCTION}, store=Store(tmp_path / "s"), workdir=tmp_path / "w", pack_root=PACK_ROOT, backend="agentenv")
    for r in responders():
        w.attach(r)
    await w.start()
    try:
        res = await run_episode(w, {"agent": ScriptedPolicy("agent", oracle())}, max_steps=40)
        assert res.done and not res.masked
        await _check_expert_end_state(w)
    finally:
        await w.close()


async def test_toy_episode_exports_valid_atif(tmp_path: Path):
    from harbor.models.trajectories import Trajectory

    from crucible.interop.harbor.atif import episode_to_atif

    w = World(helpdesk_spec(), instructions={"agent": INSTRUCTION}, store=Store(tmp_path / "s"), workdir=tmp_path / "w", pack_root=PACK_ROOT)
    for r in responders():
        w.attach(r)
    await w.start()
    try:
        res = await run_episode(w, {"agent": ScriptedPolicy("agent", oracle())}, max_steps=40)
        traj = episode_to_atif(res.trace, "agent", instruction=INSTRUCTION, model_name="oracle")
        again = Trajectory.model_validate_json(traj.model_dump_json())  # Harbor's validators: step ids, call ids
        tool_steps = [s for s in again.steps if s.tool_calls]
        assert len(tool_steps) == 11 and all(s.observation and s.observation.results for s in tool_steps)
        assert any(s.source == "user" and "crack along the base" in str(s.message) for s in again.steps)
        assert again.schema_version.startswith("ATIF-v1")
    finally:
        await w.close()


async def test_same_episode_same_state_on_local_and_agentenv(tmp_path: Path):
    """Backend independence: the oracle episode ends in identical App fingerprints on both backends."""
    pytest.importorskip("agent_env")
    fps = {}
    for backend in ("local", "agentenv"):
        w = World(helpdesk_spec(), instructions={"agent": INSTRUCTION}, store=Store(tmp_path / f"s-{backend}"), workdir=tmp_path / f"w-{backend}",
                  pack_root=PACK_ROOT, backend=backend)
        for r in responders():
            w.attach(r)
        await w.start()
        try:
            res = await run_episode(w, {"agent": ScriptedPolicy("agent", oracle())}, max_steps=40)
            fps[backend] = {a: ref.fingerprint for a, ref in res.final.apps.items()}
        finally:
            await w.close()
    assert fps["local"] == fps["agentenv"]
