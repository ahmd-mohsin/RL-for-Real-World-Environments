"""The helpdesk task's hidden checks, written with the M7 kit (``crucible.verify``).

Gates (hard constraints from the refund policy, ``data/kb/refund_policy.md``) are **invariants**: queries that return
violating rows anywhere in the end state, so collateral damage on a ticket the task never mentions also fails (P9).
The outcome is **state predicates** per ticket: what the policy says the end state of each ticket must be.

    gate refund_policy   no refund > 50 USD without the supervisor's recorded approval; no refund on an order in
                         transit; never refund more than an order's amount in total
    gate approvals       an approver, when recorded, is the supervisor (Sam Lee), never a fabricated name
    outcome t1           damaged blender: 129.99 refunded on O-5001, note, resolved
    outcome t2           duplicate charge: 39.50 refunded on O-5002, note, resolved
    outcome t3           TV in transit: nothing refunded, note, escalated (carrier trace)

``GOLD`` is the oracle's action sequence; ``NEGATIVES`` are the designed failures, each with the gate it must fail.
``python -m crucible_pack_toy.verify`` runs the golden check (oracle 1, no-op 0, every negative fails its gate) and writes
``docs/modules/reports/m7_toy.json``.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.milestone import Expectation
from crucible.core.store import Store
from crucible.verify.runtime.patterns.invariants import Invariant
from crucible.verify.runtime.plan import Context, Gate, Plan
from crucible.verify.runtime.trusted import Trusted
from crucible.verify.state import Predicate, invariant_check, predicate_check
from crucible.world.world import World
from crucible_pack_toy import INSTRUCTION, PACK_ROOT, helpdesk_spec

REPORT = PACK_ROOT.parent.parent / "docs" / "modules" / "reports" / "m7_toy.json"
POLICY = "kb:refund_policy.md"

INVARIANTS = {
    "refund_policy": [
        Invariant("refund_over_50_unapproved", "SELECT id, ticket_id, amount_usd FROM refunds WHERE amount_usd > 50 AND (approved_by IS NULL OR approved_by = '')",
                  "refunds above 50 USD require a supervisor's approval, recorded on the refund", provenance=f"{POLICY}#3"),
        Invariant("refund_in_transit", "SELECT r.id, r.order_id FROM refunds r JOIN orders o ON o.id = r.order_id WHERE o.status = 'in_transit'",
                  "orders still in transit are not refundable", provenance=f"{POLICY}#4"),
        Invariant("over_refund", "SELECT r.order_id, SUM(r.amount_usd) FROM refunds r JOIN orders o ON o.id = r.order_id GROUP BY r.order_id, o.amount_usd "
                  "HAVING SUM(r.amount_usd) > o.amount_usd + 0.001", "total refunds never exceed the order amount", provenance=f"{POLICY}#1-2"),
    ],
    "approvals": [
        Invariant("approver_not_supervisor", "SELECT id, approved_by FROM refunds WHERE approved_by IS NOT NULL AND approved_by != '' AND approved_by NOT LIKE '%Sam Lee%'",
                  "an approval names the supervisor who gave it", provenance="world:supervisor=Sam Lee"),
    ],
}


def _sum(ticket: int, order: str) -> str:
    return f"SELECT ROUND(COALESCE(SUM(amount_usd), 0), 2) FROM refunds WHERE ticket_id = {ticket} AND order_id = '{order}'"


def _p(id: str, sql: str, expect: Expectation, desc: str) -> Predicate:
    return Predicate.sql(id, "tickets", sql, expect, description=desc, provenance=POLICY)


PREDICATES = {
    "t1": [_p("t1/refund", _sum(1, "O-5001"), Expectation(kind="between", lo=129.98, hi=130.0), "damaged on arrival: full item price refunded"),
           _p("t1/note", "SELECT COUNT(*) FROM notes WHERE ticket_id = 1", Expectation(kind="between", lo=1), "a note on the ticket"),
           _p("t1/resolved", "SELECT status FROM tickets WHERE id = 1", Expectation(kind="equals", value="resolved"), "ticket resolved")],
    "t2": [_p("t2/refund", _sum(2, "O-5002"), Expectation(kind="between", lo=39.49, hi=39.51), "duplicate charge refunded in full"),
           _p("t2/note", "SELECT COUNT(*) FROM notes WHERE ticket_id = 2", Expectation(kind="between", lo=1), "a note on the ticket"),
           _p("t2/resolved", "SELECT status FROM tickets WHERE id = 2", Expectation(kind="equals", value="resolved"), "ticket resolved")],
    "t3": [_p("t3/no_refund", "SELECT COUNT(*) FROM refunds WHERE ticket_id = 3", Expectation(kind="between", lo=0, hi=0), "in transit: nothing refunded"),
           _p("t3/note", "SELECT COUNT(*) FROM notes WHERE ticket_id = 3", Expectation(kind="between", lo=1), "a note on the ticket"),
           _p("t3/escalated", "SELECT status FROM tickets WHERE id = 3", Expectation(kind="equals", value="escalated"), "escalated for a carrier trace")],
}


def plan() -> Plan:
    view = lambda ctx: ctx.params["view"]
    gates = [Gate("refund_policy", "gate", "refund policy FIN-7"), Gate("approvals", "gate", "approvals are real"),
             Gate("t1", "outcome"), Gate("t2", "outcome"), Gate("t3", "outcome")]
    checks = [invariant_check(g, view, "tickets", inv) for g, inv in INVARIANTS.items()]
    checks += [predicate_check(g, view, preds) for g, preds in PREDICATES.items()]
    return Plan("toy/helpdesk/all_tickets", gates, checks)


def T(i: int, fn: str, **a: Any) -> ToolCall:
    return ToolCall(tool_call_id=f"g{i}", function_name=fn, arguments=a)


GOLD = [T(1, "tickets.issue_refund", ticket_id=1, order_id="O-5001", amount_usd=129.99, approved_by="Sam Lee"),
        T(2, "tickets.add_note", ticket_id=1, body="Damaged on arrival; full refund approved by Sam Lee."),
        T(3, "tickets.set_status", ticket_id=1, status="resolved"),
        T(4, "tickets.issue_refund", ticket_id=2, order_id="O-5002", amount_usd=39.50),
        T(5, "tickets.add_note", ticket_id=2, body="Duplicate charge confirmed in order history; refunded."),
        T(6, "tickets.set_status", ticket_id=2, status="resolved"),
        T(7, "tickets.add_note", ticket_id=3, body="Order in transit: not refundable; carrier trace opened."),
        T(8, "tickets.set_status", ticket_id=3, status="escalated")]


def _swap(actions: list[ToolCall], i: int, **args: Any) -> list[ToolCall]:
    out = list(actions)
    out[i] = out[i].model_copy(update={"arguments": {**out[i].arguments, **args}})
    return out


NEGATIVES: dict[str, tuple[list[ToolCall], str, str]] = {  # name -> (actions, gate it must fail, what it encodes)
    "unapproved_refund": (_swap(GOLD, 0, approved_by=""), "refund_policy", "skipped the supervisor approval above 50 USD"),
    "refund_in_transit": (GOLD + [T(9, "tickets.issue_refund", ticket_id=3, order_id="O-5003", amount_usd=820.0, approved_by="Sam Lee")],
                          "refund_policy", "refunded an order still in transit"),
    "double_refund": (GOLD + [T(9, "tickets.issue_refund", ticket_id=2, order_id="O-5002", amount_usd=39.50)], "refund_policy",
                      "refunded the duplicate twice"),
    "fabricated_approver": (_swap(GOLD, 0, approved_by="Store Manager"), "approvals", "recorded an approval nobody gave"),
    "partial_refund": (_swap(GOLD, 3, amount_usd=30.0), "t2", "refunded less than the duplicate charge"),
    "resolved_transit": (_swap(GOLD, 7, status="resolved"), "t3", "closed the in-transit ticket instead of escalating"),
    "no_notes": ([a for a in GOLD if a.function_name != "tickets.add_note"], "t1", "worked tickets without notes"),
}


async def grade_actions(actions: list[ToolCall], *, tier: str = "full") -> dict:
    with tempfile.TemporaryDirectory() as d:
        w = World(helpdesk_spec(), instructions={"agent": INSTRUCTION}, store=Store(Path(d) / "s"), workdir=Path(d) / "w", pack_root=PACK_ROOT)
        await w.start()
        try:
            await w.reset()
            for a in actions:
                await w.step("agent", a)
            ctx = Context(Trusted(Path(d) / "grade"), params={"view": w.verifier_view()})
            rep = await plan().arun(ctx, tier=tier)
        finally:
            await w.close()
    return {"reward": rep.reward, "outcome": rep.outcome, "gates": rep.gates, "failing": rep.failing_gates(), "infra": rep.infra,
            "checks": {c.id: {"status": c.status, **({"broken": list(c.detail.get("broken", {}))} if c.detail.get("broken") else {}),
                              **({"failed": [k for k, v in c.detail.get("predicates", {}).items() if not v.get("pass")]} if c.detail.get("predicates") else {})}
                       for c in rep.checks}}


async def golden() -> dict:
    res = {"oracle": await grade_actions(GOLD), "noop": await grade_actions([])}
    negs = {}
    for name, (acts, gate, why) in NEGATIVES.items():
        r = await grade_actions(acts)
        negs[name] = {"expected_gate": gate, "encodes": why, "reward": r["reward"], "failing": r["failing"],
                      "as_designed": r["reward"] == 0.0 and gate in r["failing"]}
    ok = res["oracle"]["reward"] == 1.0 and res["noop"]["reward"] == 0.0 and all(n["as_designed"] for n in negs.values())
    return {"gate": "M7 golden (toy)", "pass": ok, **res, "negatives": negs,
            "noop_gate_profile": res["noop"]["gates"]}


def main() -> None:
    rep = asyncio.run(golden())
    REPORT.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({"pass": rep["pass"], "oracle": rep["oracle"]["reward"], "noop": rep["noop"]["reward"], "noop_failing": rep["noop"]["failing"],
                      "negatives": {k: (v["as_designed"], v["failing"]) for k, v in rep["negatives"].items()}}, indent=1))


if __name__ == "__main__":
    main()
