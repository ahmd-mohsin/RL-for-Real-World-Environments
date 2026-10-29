#!/usr/bin/env python
"""Smoke: the World's simulated parts end to end on the CPU, no model, no Docker, no network.

    python scripts/smoke/world_sims.py --out <dir>

1. **helpdesk** (the toy pack's World: a SQLite system of record with a tools module, a document library, a mailbox,
   grounded customers, a rule-based supervisor, a step clock and three events): a scripted agent lists tickets, asks a
   customer (the grounded reply quotes the customer's record), asks the supervisor (rule reply), issues a refund (the
   ``on_tool`` trigger mails an audit note), then waits out the clock: the 1 h reminder fires and the 3 h permission
   revocation makes the next refund a refused call. Snapshot -> mutate -> restore gives back the snapshot's rows, and
   the verifier view reads the database.
2. **website**: the toy vendor portal (an ASGI app under uvicorn over its SQLite DB) through ``http_request`` (the
   browser App needs ``npx @playwright/mcp``, a download: not run here); its database read through the verifier view.
   The same episode runs again on the **agentenv backend** (Crucible Apps behind agentenv's gateway process: its role
   ACLs, its MCP endpoint); the two runs must agree, and an MCP client with an ``AgentEnv-Role`` header calls a tool.
3. **json_api**: create / update / list on a JSON-document service; snapshot and restore.

Exit 1 when a check fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def rows(r: Any) -> list:
    """The rows of a verifier-view SQL result (``value.rows``); a failed query raises."""
    if not getattr(r, "ok", False):
        raise RuntimeError(f"verifier query failed: {getattr(r, 'error', r)}")
    return list((r.value or {}).get("rows") or [])


def tc(i: str, fn: str, **args: Any) -> Any:
    from crucible.core.schema.actions import ToolCall

    return ToolCall(tool_call_id=i, function_name=fn, arguments=args)


async def helpdesk(out: Path, backend: str = "local") -> dict[str, Any]:
    from crucible_pack_toy import INSTRUCTION, PACK_ROOT, helpdesk_spec, responders

    from crucible.actors.policy import ScriptedPolicy, run_episode
    from crucible.core.schema.actions import Message
    from crucible.core.store import Store
    from crucible.world.world import World

    spec = helpdesk_spec()
    if backend != "local":
        spec = spec.model_copy(update={"backend": backend})
    w = World(spec, instructions={"agent": INSTRUCTION}, pack_root=PACK_ROOT, store=Store(out / "s"), workdir=out / "w")
    for r in responders():
        w.attach(r)
    await w.start()
    res: dict[str, Any] = {}
    try:
        script = [tc("1", "tickets.list_tickets"),
                  Message(to="customer_lee", text="How much were you charged for O-5002?"),
                  tc("2", "tickets.request_approval", ticket_id=1, order_id="O-5001", amount_usd=60.0, reason="cracked on arrival"),
                  Message(to="supervisor", text="Please approve a refund of 60 USD for ticket 1, order O-5001."),
                  tc("3", "tickets.issue_refund", ticket_id=1, order_id="O-5001", amount_usd=60.0, approved_by="Sam Lee"),
                  *[tc(f"r{i}", "kb.list_docs") for i in range(40)],  # 40 steps x 300 s: past the 1 h and 3 h events
                  tc("4", "tickets.issue_refund", ticket_id=2, order_id="O-5002", amount_usd=10.0)]
        ep = await run_episode(w, {"agent": ScriptedPolicy("agent", script)}, max_steps=200)
        ev = ep.trace.events
        lee = next((e.text for e in ev if e.actor == "customer_lee" and e.text), "")
        sup = next((e.text for e in ev if e.actor == "supervisor" and e.text), "")
        fired = [f["event"] for f in w.firing_log()]
        view = w.verifier_view()
        q = "SELECT ticket_id, order_id, amount_usd FROM refunds ORDER BY id"
        refunds = rows(await view.query("tickets", {"op": "sql", "args": {"sql": q}}))
        inbox = rows(await view.query("mail", {"op": "sql", "args": {"sql": "SELECT subject FROM messages"}}))
        res.update(steps=ep.steps, stop=ep.stop_condition, clock=w.clock.now_iso(), lee_reply=lee[:160], supervisor_reply=sup[:160], fired=fired,
                   refunds=refunds, mail_subjects=inbox)
        # snapshot -> mutate -> restore (the agent declared done: the mutation is a system call, as a trigger's would be)
        nq = {"op": "sql", "args": {"sql": "SELECT count(*) AS n FROM notes"}}
        ref = await w.snapshot()
        notes_before = rows(await view.query("tickets", nq))
        await w._call_tool("agent", tc("5", "tickets.add_note", ticket_id=3, body="smoke note"), system=True)
        notes_after = rows(await view.query("tickets", nq))
        await w.restore(ref)
        notes_back = rows(await w.verifier_view().query("tickets", nq))
        res.update(notes_at_snapshot=notes_before, notes_after_mutation=notes_after, notes_after_restore=notes_back)
        if backend == "agentenv":  # any MCP harness: the gateway's aggregate endpoint, the role from the header
            from crucible.world.packaging.sidecar import McpClient

            agent = McpClient(w.backend.mcp_url, headers={"AgentEnv-Role": "agent"})
            names = {t.name for t in await agent.list_tools("gw")}
            r = await agent.call("gw", "tickets__get_ticket", {"ticket_id": 3}, call_id="mcp1")
            res["mcp"] = {"tools": len(names), "has_tickets": "tickets__get_ticket" in names, "call_ok": r.ok, "call": str(r.content)[:160]}
    finally:
        await w.close()
    res["checks"] = {
        "grounded_customer_replied": bool(lee),
        "supervisor_replied": bool(sup),
        "on_tool_trigger_fired": "refund_audit" in fired,
        "audit_mail_delivered": any("refund logged" in str(r) for r in inbox),
        "time_event_fired": "sla_reminder" in fired,
        "permission_revoked": "month_end_freeze" in fired,
        "late_refund_refused": len(refunds) == 1,
        "restore_exact": notes_back == notes_before and notes_after != notes_before,
        **({"mcp_harness": bool(res["mcp"]["has_tickets"] and res["mcp"]["call_ok"])} if "mcp" in res else {}),
    }
    res["ok"] = all(res["checks"].values())
    return res


async def website(out: Path) -> dict[str, Any]:
    from crucible_pack_toy import PACK_ROOT

    from crucible.core.schema.world import WorldSpec
    from crucible.core.store import Store
    from crucible.world.world import World

    spec = WorldSpec.model_validate({
        "world": "smoke/vendor-portal",
        "apps": [{"id": "portal", "template": "common/website", "state": {"adapter": "sql", "params": {"path": "portal/portal.db"}},
                  "params": {"app": "data/portal/site.py:create_app", "api_tools": True}}],
        "actors": [{"id": "clerk", "kind": "policy", "tools": ["portal.*"]}]})
    w = World(spec, store=Store(out / "s"), workdir=out / "w", pack_root=PACK_ROOT)
    await w.start()
    res: dict[str, Any] = {}
    try:
        await w.reset()
        r = await w.step("clerk", tc("1", "portal.http_request", method="GET", path="/"))
        page = r.observation.tool_results[0]
        tables = rows(await w.verifier_view().query("portal", {"op": "sql", "args": {"sql": "SELECT name FROM sqlite_master WHERE type='table'"}}))
        res.update(home_ok=page.ok, home=str(page.content)[:200], tables=tables)
    finally:
        await w.close()
    res["ok"] = bool(res.get("home_ok")) and "HTTP 200" in res.get("home", "")
    return res


async def json_api(out: Path) -> dict[str, Any]:
    from crucible.core.schema.world import WorldSpec
    from crucible.core.store import Store
    from crucible.world.world import World

    spec = WorldSpec.model_validate({
        "world": "smoke/crm", "apps": [{"id": "crm", "template": "common/json_api", "state": {"adapter": "json_doc", "params": {"path": "crm.json"}},
                                         "params": {"collections": ["accounts"]}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["crm.*"]}]})
    w = World(spec, store=Store(out / "s"), workdir=out / "w")
    await w.start()
    try:
        await w.reset()
        a = (await w.step("agent", tc("1", "crm.create_resource", collection="accounts", id="A1", data={"name": "Acme", "tier": "gold"}))).observation.tool_results[0]
        ref = await w.snapshot()
        u = (await w.step("agent", tc("2", "crm.update_resource", collection="accounts", id="A1", patch={"tier": "standard"}))).observation.tool_results[0]
        await w.restore(ref)
        g = (await w.step("agent", tc("3", "crm.get_resource", collection="accounts", id="A1"))).observation.tool_results[0]
    finally:
        await w.close()
    res = {"create_ok": a.ok, "update_ok": u.ok, "after_restore": str(g.content)[:200]}
    res["ok"] = a.ok and u.ok and '"gold"' in res["after_restore"]
    return res


async def main_async(out: Path) -> dict[str, Any]:
    res: dict[str, Any] = {}
    parts = (("helpdesk", helpdesk), ("helpdesk_agentenv", lambda d: helpdesk(d, backend="agentenv")), ("website", website), ("json_api", json_api))
    for name, fn in parts:
        d = out / name
        d.mkdir(parents=True)
        try:
            res[name] = await fn(d)
        except Exception as e:  # noqa: BLE001 - one part's failure is reported, the others still run
            res[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:1000]}
    a, b = res.get("helpdesk") or {}, res.get("helpdesk_agentenv") or {}
    keys = ("fired", "refunds", "mail_subjects", "lee_reply", "supervisor_reply", "steps")
    res["backend_parity"] = {"ok": all(a.get(k) == b.get(k) for k in keys), "differs": [k for k in keys if a.get(k) != b.get(k)]}
    res["ok"] = all(v.get("ok") for v in res.values() if isinstance(v, dict))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    res = asyncio.run(main_async(out))
    print(json.dumps(res, indent=1, default=str))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
