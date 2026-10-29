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


async def test_external_mcp_harness_drives_world_through_gateway(tmp_path: Path):
    """Any MCP harness (Claude Code, Codex, ...) can work a Crucible World: connect to the gateway's
    aggregate MCP endpoint with an AgentEnv-Role header; agentenv enforces that role's tool access."""
    from crucible.world.packaging.sidecar import McpClient

    w = World(spec("step"), store=Store(tmp_path / "store"), workdir=tmp_path / "w")
    await w.start()
    try:
        await w.reset()
        url = w.backend.mcp_url
        clerk = McpClient(url, headers={"AgentEnv-Role": "clerk"})
        auditor = McpClient(url, headers={"AgentEnv-Role": "auditor"})
        clerk_tools = {t.name for t in await clerk.list_tools("gw")}
        auditor_tools = {t.name for t in await auditor.list_tools("gw")}
        assert {"erp__create_resource", "mail__send_email"} <= clerk_tools
        assert auditor_tools == {"erp__list_resources", "erp__get_resource"}
        r = await clerk.call("gw", "erp__create_resource", {"collection": "invoices", "id": "INV-42", "data": {"amount": 7}}, call_id="x1")
        assert r.ok, r.content
        denied = await auditor.call("gw", "erp__create_resource", {"collection": "invoices", "id": "INV-43", "data": {}}, call_id="x2")
        assert not denied.ok
        assert (await w.verifier_view().query("erp", {"op": "get", "args": {"path": "/invoices"}})).value == {"INV-42": {"amount": 7}}
    finally:
        await w.close()


async def test_agentenv_restore_keeps_fired_triggers_fired(tmp_path: Path):
    """Snapshots taken after a one-shot time trigger fired restore it as fired (no second vendor reply), and the
    permission trigger that is still due fires at its absolute instant after a restore."""
    w = World(spec("wall"), store=Store(tmp_path / "store"), workdir=tmp_path / "w")
    await w.start()
    try:
        await w.reset()
        assert {"vendor_reply", "freeze"} <= w.backend.delegated_events()
        await _until(lambda: any("CN-7" in m.text for m in w.inboxes["clerk"]), 8)  # fired at t0 + 1h (1 s real)
        w.inboxes["clerk"].clear()
        ref = await w.snapshot()
        fired = {t["id"]: t["fire_count"] for t in ref.backend["triggers"]["triggers"]}
        assert fired["vendor_reply"] == 1
        await w.restore(ref)
        state = (await w.backend.http.get("/triggers/state")).json()
        ids = {t["id"] for t in state["triggers"]}
        one_shots = {"vendor_reply", "freeze"}
        assert ids == {i for i in fired if not (i in one_shots and fired[i])}  # fired one-shots stay fired, the rest stay armed
        await asyncio.sleep(1.5)
        assert not any("CN-7" in m.text for m in w.inboxes["clerk"])  # no duplicate delivery after the restore
        if not fired["freeze"]:
            await _until_async(lambda: _lacks(w, "clerk", "erp.create_resource"), 8)  # still due: fires at t0 + 2h
    finally:
        await w.close()


async def _lacks(w: World, actor: str, tool: str) -> bool:
    await w.backend._refresh_allowed(w)
    return tool not in {t.qualified for t in w.backend.allowed_tools(actor)}


@pytest.mark.parametrize("backend", ["local", "agentenv"])
async def test_a_permission_event_the_worlds_scheduler_fires_is_enforced_on_every_backend(tmp_path: Path, backend: str):
    """Smoke 2026-10-29: under a step clock the World's scheduler fires the freeze (not delegated). It revoked the tool
    in ``World.acl`` only; the agentenv gateway checks access itself (``native_acl``), so the clerk kept creating
    resources there while the local backend refused them. Both must refuse after the freeze, and hide the tool."""
    d = spec("step").model_dump()
    d["backend"] = backend
    d["clock"]["step_dt_s"] = 3600  # one step = one virtual hour: the 2 h freeze fires after the second step
    w = World(WorldSpec.model_validate(d), store=Store(tmp_path / "store"), workdir=tmp_path / "w")
    await w.start()
    try:
        await w.reset()
        for i in range(2):
            r = await w.step("clerk", ToolCall(tool_call_id=f"l{i}", function_name="erp.list_resources", arguments={"collection": "invoices"}))
            assert r.observation.tool_results[0].ok
        assert "freeze" in [f["event"] for f in w.firing_log()]
        r = await w.step("clerk", ToolCall(tool_call_id="c", function_name="erp.create_resource", arguments={"collection": "invoices", "id": "INV-1", "data": {}}))
        res = r.observation.tool_results[0]
        assert not res.ok and res.error == "permission_denied", res.content
        assert "erp.create_resource" not in {t.qualified for t in w.tools_for("clerk")}
        assert (await w.verifier_view().query("erp", {"op": "get", "args": {"path": "/invoices"}})).value in ({}, None)
    finally:
        await w.close()
