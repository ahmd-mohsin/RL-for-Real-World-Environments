"""M1 agentenv backend: Crucible Apps behind a real agentenv gateway process.

Proves we use agentenv's own machinery: role-based tool access (403 from the gateway itself), the
aggregate MCP endpoint, the virtual clock + get_time tool, time / action triggers with tool,
permission and nl actions (nl delivered through an agentenv-protocol A2A relay agent).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from crucible.core.schema.actions import Message, ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import World

pytest.importorskip("agent_env")


def spec(clock_mode: str = "wall") -> WorldSpec:
    return WorldSpec.model_validate(
        {
            "world": "test/agentenv-helpdesk",
            "backend": "agentenv",
            "apps": [
                {"id": "erp", "template": "common/json_api", "state": {"adapter": "json_doc", "params": {"path": "erp.json"}},
                 "params": {"collections": ["invoices", "payments"]}},
                {"id": "mail", "template": "common/mailbox", "state": {"adapter": "sql", "params": {"path": "mail.db"}}},
            ],
            "actors": [
                {"id": "clerk", "kind": "policy", "tools": ["erp.*", "mail.*"]},
                {"id": "auditor", "kind": "policy", "tools": ["erp.list_resources", "erp.get_resource"]},
                {"id": "vendor", "kind": "simulated", "tools": ["mail.send_email"]},
            ],
            "clock": {"mode": clock_mode, "t0": "2026-03-02T09:00:00Z", "rate": 3600, "step_dt_s": 60},
            "events": [
                # vendor writes after 1 virtual hour (= 1 real second at rate 3600)
                {"id": "vendor_reply", "when": {"after": "1h"}, "actor": "vendor", "action": {"type": "nl", "to": "clerk", "text": "Credit note CN-7 issued for INV-1."}},
                # every payment creation sends an audit email (action trigger with a tool action)
                {"id": "audit_mail", "when": {"on_tool": "erp.create_resource"}, "actor": "vendor",
                 "action": {"type": "tool", "tool": "mail.send_email", "args": {"to": ["clerk@world.local"], "subject": "payment audit", "body": "logged"}}},
                # month-end freeze after 2 virtual hours: the clerk can no longer create resources
                {"id": "freeze", "when": {"after": "2h"}, "actor": "vendor", "action": {"type": "permission", "to": "clerk", "revoke": ["erp.create_resource"]}},
            ],
        }
    )


async def _until(cond, timeout_s: float) -> None:
    for _ in range(int(timeout_s / 0.1)):
        if cond():
            return
        await asyncio.sleep(0.1)
    raise AssertionError("condition not met in time")


async def _until_async(cond, timeout_s: float) -> None:
    for _ in range(int(timeout_s / 0.1)):
        if await cond():
            return
        await asyncio.sleep(0.1)
    raise AssertionError("condition not met in time")


async def test_agentenv_gateway_world(tmp_path: Path):
    w = World(spec(), store=Store(tmp_path / "store"), workdir=tmp_path / "w", instructions={"clerk": "Pay INV-1 unless a credit note arrives."})
    await w.start()
    try:
        b = w.backend
        assert b.delegated_events() == {"vendor_reply", "audit_mail", "freeze"}
        obs = await w.reset()
        assert {t.qualified for t in obs["clerk"].tools} >= {"erp.create_resource", "mail.list_inbox"}
        assert {t.qualified for t in obs["auditor"].tools} == {"erp.list_resources", "erp.get_resource"}
        # role ACL is enforced by the gateway itself (raw REST call with the auditor role -> 403)
        async with httpx.AsyncClient(base_url=b.gateway_url) as c:
            r = await c.post("/step", headers={"AgentEnv-Role": "auditor"}, json={"action": "call_tool", "tool_name": "erp__create_resource", "arguments": {}})
            assert r.status_code == 403
            r = await c.post("/step", json={"action": "list_tools"})  # anonymous clients get nothing
            assert r.json()["tools"] == []
            st = (await c.get("/clock/state")).json()
            assert st["armed"] and st["virtual_seconds_per_real_second"] == 3600
        r = await w.step("auditor", ToolCall(tool_call_id="a", function_name="erp.create_resource", arguments={"collection": "payments", "id": "P-0", "data": {}}))
        assert r.observation.tool_results[0].error == "permission_denied"
        # action trigger: creating a payment makes the gateway send the audit email via our mailbox App
        r = await w.step("clerk", ToolCall(tool_call_id="1", function_name="erp.create_resource", arguments={"collection": "payments", "id": "P-1", "data": {"invoice": "INV-1", "amount": 100}}))
        assert r.observation.tool_results[0].ok, r.observation.tool_results[0].content
        r = await w.step("clerk", ToolCall(tool_call_id="2", function_name="mail.list_inbox", arguments={}))
        assert "payment audit" in r.observation.tool_results[0].content
        # time trigger (nl): > 1 virtual hour later the gateway's driver fires it and calls our A2A relay
        await _until(lambda: any("CN-7" in m.text for m in w.inboxes["clerk"]), 8)
        r = await w.step("clerk", ToolCall(tool_call_id="3", function_name="erp.list_resources", arguments={"collection": "payments"}))
        assert any("CN-7" in m.text for m in r.observation.inbox), r.observation.inbox
        gw_time = await b.gateway_time()
        assert gw_time and gw_time > "2026-03-02T10:00:00"
        # permission trigger after 2 virtual hours: create_resource disappears for the clerk
        async def frozen() -> bool:
            r = await b.http.post("/step", headers={"AgentEnv-Role": "clerk"}, json={"action": "list_tools"})
            return "erp__create_resource" not in {t["name"] for t in r.json()["tools"]}

        await _until_async(frozen, 8)
        r = await w.step("clerk", ToolCall(tool_call_id="4", function_name="erp.create_resource", arguments={"collection": "payments", "id": "P-2", "data": {}}))
        assert r.observation.tool_results[0].error == "permission_denied"
        r = await w.step("clerk", Message(to="auditor", text="frozen"))
        view = w.verifier_view()
        assert (await view.query("erp", {"op": "get", "args": {"path": "/payments"}})).value == {"P-1": {"invoice": "INV-1", "amount": 100}}
        trace = w.to_trace()
        assert any(e.type == "event:nl" and e.data.get("via") == "agentenv-trigger" for e in trace.events)
        traj = (tmp_path / "w" / "gateway_trajectory.jsonl").read_text()
        assert "erp__create_resource" in traj  # agentenv's own trajectory log recorded the calls
    finally:
        await w.close()


async def test_agentenv_snapshot_restore_roles_and_state(tmp_path: Path):
    w = World(spec("step"), store=Store(tmp_path / "store"), workdir=tmp_path / "w")
    await w.start()
    try:
        # step mode: every event stays in Crucible's deterministic, restorable scheduler; the gateway still
        # routes every tool call and enforces role access.
        assert w.backend.delegated_events() == set()
        await w.reset()
        await w.step("clerk", ToolCall(tool_call_id="1", function_name="erp.create_resource", arguments={"collection": "invoices", "id": "INV-9", "data": {"amount": 5}}))
        r = await w.step("clerk", ToolCall(tool_call_id="1b", function_name="mail.list_inbox", arguments={}))
        assert "payment audit" in r.observation.tool_results[0].content  # on_tool event via Crucible's scheduler
        ref = await w.snapshot()
        await w.step("clerk", ToolCall(tool_call_id="2", function_name="erp.delete_resource", arguments={"collection": "invoices", "id": "INV-9"}))
        await w.restore(ref)
        assert (await w.snapshot()).fingerprint == ref.fingerprint
        r = await w.step("clerk", ToolCall(tool_call_id="3", function_name="erp.get_resource", arguments={"collection": "invoices", "id": "INV-9"}))
        assert '"amount": 5' in r.observation.tool_results[0].content
    finally:
        await w.close()
