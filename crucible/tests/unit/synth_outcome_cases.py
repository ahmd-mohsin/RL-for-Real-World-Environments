"""Known-label cases for the outcome classifier (``synth.outcome``): the fixture tasks, the planted defects of
``synth_fixtures.planted`` and scripted student attempts whose intended or unintended outcome is known by construction.
Framework fixtures, not training tasks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from crucible.synth.falsify import FalsifyContext
from crucible.synth.learn import AgentStudentProbe, ContextPool, Rollout
from crucible.synth.task import Task
from tests.unit import synth_fixtures as fx

FIND = ("ap.find_invoice", {"number": "INV-1003"})
PO = ("ap.get_purchase_order", {"po_id": "PO-3"})
HOLD = ("ap.hold_invoice", {"invoice_id": 103, "reason": "price_variance"})


def _t3_calls() -> list[tuple[str, dict[str, Any]]]:
    return [("desk.get_ticket", {"ticket_id": "T-7"}), ("desk.search_kb", {"query": "password reset"}),
            ("desk.resolve_ticket", {"ticket_id": "T-7", "resolution": "KB-3"})]


# (name, task factory, plan, expected label). A plan is "oracle", "noop" or (calls, answer).
CASES: list[tuple[str, Callable[[], Task], Any, str]] = [
    ("clean t1: the oracle's attempt", fx.t1_hold, "oracle", "intended_pass"),
    ("clean t1: approves instead of holding", fx.t1_hold, ([FIND, PO, ("ap.approve_invoice", {"invoice_id": 103})], None), "intended_fail"),
    ("clean t1: does nothing", fx.t1_hold, "noop", "intended_fail"),
    ("clean t1: holds with the wrong reason", fx.t1_hold, ([FIND, PO, ("ap.hold_invoice", {"invoice_id": 103, "reason": "po_mismatch"})], None),
     "intended_fail"),
    ("clean t2: the oracle's attempt", fx.t2_approve_answer, "oracle", "intended_pass"),
    ("clean t2: a wrong figure", fx.t2_approve_answer,
     ([("ap.find_invoice", {"number": "INV-1001"}), ("ap.get_receipts", {"po_id": "PO-1"}), ("ap.approve_invoice", {"invoice_id": 101})], 999.0),
     "intended_fail"),
    ("clean t3: the oracle's attempt", fx.t3_resolve_email, "oracle", "intended_pass"),
    ("clean t3: never emails the customer", fx.t3_resolve_email, (_t3_calls(), None), "intended_fail"),
    ("clean t4: the oracle's attempt", fx.t4_priority, "oracle", "intended_pass"),
    ("clean t4: one of the two tickets", fx.t4_priority,
     ([("desk.list_tickets", {"status": "open"}), ("desk.set_priority", {"ticket_id": "T-2", "priority": "high"})], None), "intended_fail"),
    ("clean t5: stops after one decision", fx.t5_batch, ([("ap.find_invoice", {"number": "INV-1002"}), ("ap.approve_invoice", {"invoice_id": 102})], None),
     "intended_fail"),
    ("clean t6: the oracle's attempt", fx.t6_resolve, "oracle", "intended_pass"),
    ("D7 leak: holds without the purchase order", lambda: fx.planted("D7"), ([FIND, HOLD], None), "unintended_pass"),
    ("D5 hidden: cannot tell which invoice", lambda: fx.planted("D5"), "noop", "unintended_fail"),
    ("D5 hidden: holds another invoice", lambda: fx.planted("D5"),
     ([("ap.list_invoices", {"status": "pending"}), ("ap.hold_invoice", {"invoice_id": 104, "reason": "price_variance"})], None), "unintended_fail"),
    ("D6 over-strict: holds before reading the PO (valid)", lambda: fx.planted("D6"), ([FIND, HOLD, PO], None), "unintended_fail"),
    ("D10 unsolvable: the oracle's attempt fails", lambda: fx.planted("D10"), "oracle", "unintended_fail"),
    ("vacuous: does nothing and passes", fx.always_pass, "noop", "unintended_pass"),
    ("D9 insensitive: holds another invoice and passes", lambda: fx.planted("D9"),
     ([("ap.find_invoice", {"number": "INV-1004"}), ("ap.hold_invoice", {"invoice_id": 104, "reason": "price_variance"})], None), "unintended_pass"),
    ("D8 affordance: the admin tool", lambda: fx.planted("D8"),
     ([("desk.admin_update", {"ticket_id": "T-3", "status": "resolved", "resolution": "KB-2"})], None), "unintended_pass"),
    ("D11 nondeterministic: the oracle's attempt", lambda: fx.planted("D11"), "oracle", "unintended_pass"),
    ("D2 frame hole: also approves another invoice", lambda: fx.planted("D2"),
     ([FIND, PO, HOLD, ("ap.approve_invoice", {"invoice_id": 104})], None), "unintended_pass"),
    ("D1 lax: resolves without the email", lambda: fx.planted("D1"), (_t3_calls(), None), "unintended_pass"),
    ("D3 lax value: a wrong figure inside the tolerance", lambda: fx.planted("D3"),
     ([("ap.find_invoice", {"number": "INV-1001"}), ("ap.get_receipts", {"po_id": "PO-1"}), ("ap.approve_invoice", {"invoice_id": 101})], 1000.0),
     "unintended_pass"),
    ("D4 vacuous pattern: the oracle's attempt (a real pass)", lambda: fx.planted("D4"), "oracle", "intended_pass"),
    ("D9 insensitive: the oracle's attempt (a real pass)", lambda: fx.planted("D9"), "oracle", "intended_pass"),
]


async def rollout_for(task: Task, plan: Any, i: int = 0) -> Rollout:
    """A scripted attempt as a graded rollout (the same path a real student's rollout takes)."""
    ctx = FalsifyContext()
    oracle = await ctx.oracle(task)

    def runner(tag: str) -> Any:
        async def go(view: Any, world: Any) -> Any:
            if plan == "oracle":
                for c in oracle.calls:
                    await world.call(c.tool, dict(c.args))
                return oracle.answer
            if plan == "noop":
                return None
            calls, ans = plan
            for tool, args in calls:
                await world.call(tool, dict(args))
            return ans

        return go

    probe = AgentStudentProbe(runner=runner, contexts=ContextPool(FalsifyContext, size=1), student="scripted", concurrency=1)
    return (await probe.rollouts(task, 1, start=i))[0]
