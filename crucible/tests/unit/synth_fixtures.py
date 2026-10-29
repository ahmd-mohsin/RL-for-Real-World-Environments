"""Test domain for ``crucible.synth`` (framework fixtures, not training tasks): two PyWorld Worlds and hand-written tasks.

- ``ap_world``: accounts payable: ``invoices``, ``purchase_orders``, ``receipts``, ``audit_log`` (every decision tool
  writes an audit row stamped with the virtual clock).
- ``helpdesk_world``: ``desk`` (tickets, kb articles) and ``mail`` (outbox). Variants plant World defects:
  ``admin`` adds a direct-write affordance (D8), ``nondet`` makes ``resolve_ticket`` stamp a random ref (D11),
  ``queue`` adds a positionally canonicalized work queue (a position-keyed contract, caught by PERT).

``CORRECT`` holds five correct tasks (oracle program + contract + intent). ``planted(code)`` returns a task carrying
exactly one defect of class D1..D11 (part2-paper-core §8's operators, applied by hand).
"""

from __future__ import annotations

import copy
import uuid
from typing import Any

from crucible.synth.canon import FieldScope, IdCanon, Normalizers
from crucible.synth.contract import AnswerCheck, Contract, EffectPattern, PAny, PApprox, PEq, TraceConstraint
from crucible.synth.iface import App, PyWorld, ToolError
from crucible.synth.program import Program, Step, fn, lit, ref
from crucible.synth.task import IntentCard, Task, WorldRef

MOD = "tests.unit.synth_fixtures"

# ---------------------------------------------------------------------------------------------------------- AP World

INVOICES = {
    101: {"id": 101, "number": "INV-1001", "vendor": "Acme Tools", "po": "PO-1", "amount": 1200.0, "status": "pending", "reason": None},
    102: {"id": 102, "number": "INV-1002", "vendor": "Borealis Freight", "po": "PO-2", "amount": 800.0, "status": "pending", "reason": None},
    103: {"id": 103, "number": "INV-1003", "vendor": "Cobalt Office", "po": "PO-3", "amount": 2600.0, "status": "pending", "reason": None},
    104: {"id": 104, "number": "INV-1004", "vendor": "Dune Logistics", "po": "PO-4", "amount": 450.0, "status": "pending", "reason": None},
    105: {"id": 105, "number": "INV-1005", "vendor": "Borealis Freight", "po": "PO-5", "amount": 300.0, "status": "pending", "reason": None},
    106: {"id": 106, "number": "INV-1006", "vendor": "Borealis Freight", "po": "PO-6", "amount": 975.5, "status": "pending", "reason": None},
    107: {"id": 107, "number": "INV-0907", "vendor": "Acme Tools", "po": "PO-7", "amount": 500.0, "status": "on_hold", "reason": "missing_receipt"},
    108: {"id": 108, "number": "INV-0908", "vendor": "Ember Foods", "po": "PO-8", "amount": 610.0, "status": "approved", "reason": None},
}
POS = {f"PO-{i}": {"id": f"PO-{i}", "vendor": v, "amount": a} for i, v, a in [
    (1, "Acme Tools", 1200.0), (2, "Borealis Freight", 800.0), (3, "Cobalt Office", 2000.0), (4, "Dune Logistics", 450.0),
    (5, "Borealis Freight", 300.0), (6, "Borealis Freight", 975.5), (7, "Acme Tools", 500.0), (8, "Ember Foods", 610.0)]}
RECEIPTS = {f"GR-{i}": {"id": f"GR-{i}", "po": po, "value": v} for i, po, v in [
    (1, "PO-1", 700.0), (2, "PO-1", 500.0), (3, "PO-2", 800.0), (4, "PO-3", 2600.0), (5, "PO-4", 450.0), (6, "PO-5", 300.0), (7, "PO-6", 500.0),
    (8, "PO-8", 610.0)]}


def _extra_rows(extra: int) -> tuple[dict, dict, dict]:
    """``extra`` more pending invoices (ids 201..), each with a matching purchase order and receipt: a World with enough
    entities that MUT's neighbourhood reaches n >= 59 distinct effects (part2-paper-core §6)."""
    inv, pos, rec = {}, {}, {}
    for i in range(extra):
        iid, po = 201 + i, f"PO-{201 + i}"
        amt = float(100 + 25 * i)
        inv[iid] = {"id": iid, "number": f"INV-2{i:03d}", "vendor": f"Vendor {i:02d}", "po": po, "amount": amt, "status": "pending", "reason": None}
        pos[po] = {"id": po, "vendor": f"Vendor {i:02d}", "amount": amt}
        rec[f"GR-{201 + i}"] = {"id": f"GR-{201 + i}", "po": po, "value": amt}
    return inv, pos, rec


def _ap_app(extra: int = 0) -> App:
    xi, xp, xr = _extra_rows(extra)
    app = App("ap", {"invoices": {**copy.deepcopy(INVOICES), **xi}, "purchase_orders": {**copy.deepcopy(POS), **xp},
                     "receipts": {**copy.deepcopy(RECEIPTS), **xr}, "audit_log": {}})

    @app.tool(read_only=True)
    def list_invoices(ctx, status: str = "pending") -> dict:
        """List the invoices with a status (pending, approved, on_hold, rejected): id, number, vendor, amount, status."""
        return {"invoices": [{k: r[k] for k in ("id", "number", "vendor", "amount", "status")} for r in ctx.tables["invoices"].values() if r["status"] == status]}

    @app.tool(read_only=True)
    def find_invoice(ctx, number: str) -> dict:
        """Every invoice with exactly this invoice number: id, number, vendor, po, amount, status, reason."""
        return {"invoices": [dict(r) for r in ctx.tables["invoices"].values() if r["number"] == number]}

    @app.tool(read_only=True)
    def get_invoice(ctx, invoice_id: int) -> dict:
        """One invoice by id."""
        r = ctx.tables["invoices"].get(invoice_id)
        if r is None:
            raise ToolError(f"no invoice {invoice_id}")
        return dict(r)

    @app.tool(read_only=True)
    def get_purchase_order(ctx, po_id: str) -> dict:
        """A purchase order: vendor and approved amount."""
        r = ctx.tables["purchase_orders"].get(po_id)
        if r is None:
            raise ToolError(f"no purchase order {po_id}")
        return dict(r)

    @app.tool(read_only=True)
    def get_receipts(ctx, po_id: str) -> dict:
        """The goods receipts recorded against a purchase order, and their total value."""
        rows = [dict(r) for r in ctx.tables["receipts"].values() if r["po"] == po_id]
        return {"po": po_id, "receipts": rows, "total": round(sum(r["value"] for r in rows), 2)}

    def decide(ctx, invoice_id: Any, status: str, reason: str | None, action: str) -> dict:
        inv = ctx.tables["invoices"].get(invoice_id)
        if inv is None:
            raise ToolError(f"no invoice {invoice_id}")
        if inv["status"] != "pending":
            raise ToolError(f"invoice {invoice_id} is {inv['status']}, not pending")
        inv["status"], inv["reason"] = status, reason
        log = ctx.tables["audit_log"]
        nid = max(log, default=0) + 1
        log[nid] = {"id": nid, "invoice_id": invoice_id, "action": action, "reason": reason, "at": ctx.now()}
        return {"id": invoice_id, "status": status, "reason": reason}

    @app.tool(writes=["invoices", "audit_log"])
    def approve_invoice(ctx, invoice_id: int) -> dict:
        """Approve a pending invoice for payment (status approved)."""
        return decide(ctx, invoice_id, "approved", None, "approve")

    @app.tool(writes=["invoices", "audit_log"])
    def hold_invoice(ctx, invoice_id: int, reason: str) -> dict:
        """Put a pending invoice on hold (status on_hold) with a reason code: price_variance (the amount exceeds the purchase
        order), missing_receipt (the goods receipts add up to less) or po_mismatch."""
        if reason not in {"price_variance", "missing_receipt", "po_mismatch"}:
            raise ToolError("unknown hold reason")
        return decide(ctx, invoice_id, "on_hold", reason, "hold")

    @app.tool(writes=["invoices", "audit_log"])
    def reject_invoice(ctx, invoice_id: int, reason: str) -> dict:
        """Reject a pending invoice (status rejected) with a reason code: duplicate or vendor_blocked."""
        if reason not in {"duplicate", "vendor_blocked"}:
            raise ToolError("unknown reject reason")
        return decide(ctx, invoice_id, "rejected", reason, "reject")

    return app


def ap_world(extra: int = 0) -> PyWorld:
    return PyWorld([_ap_app(extra)])


AP_N = Normalizers(ignore_fields=[FieldScope(app="ap", coll="audit_log", field="at")], id_canon=[IdCanon(app="ap", coll="audit_log")])

# ---------------------------------------------------------------------------------------------------------- helpdesk World

TICKETS = {t[0]: {"id": t[0], "customer": t[1], "subject": t[2], "status": t[3], "priority": t[4], "resolution": t[5]} for t in [
    ("T-1", "ann@example.com", "Invoice copy", "open", "normal", None),
    ("T-2", "bob@example.com", "Shipping delay", "open", "normal", None),
    ("T-3", "carol@example.com", "Refund request", "open", "normal", None),
    ("T-4", "dave@example.com", "Login fails", "open", "normal", None),
    ("T-5", "bob@example.com", "Wrong item", "open", "low", None),
    ("T-6", "erin@example.com", "Password expired", "resolved", "normal", "KB-3"),
    ("T-7", "ann@example.com", "Password reset", "open", "normal", None),
    ("T-8", "bob@example.com", "Address change", "resolved", "normal", "KB-1"),
]}
KB = {k: {"id": k, "title": t, "body": b} for k, t, b in [
    ("KB-1", "Change your shipping address", "Open your account settings and edit the address."),
    ("KB-2", "Refund policy", "A refund is paid within 14 days of the return."),
    ("KB-3", "How to do a password reset", "Use the link on the sign-in page; it expires in one hour."),
    ("KB-4", "Two-factor login", "Enroll an authenticator app under security."),
    ("KB-5", "Invoice copies", "Download any invoice from the billing page."),
]}
QUEUE = {f"q{i}": {"ticket": t, "state": "waiting"} for i, t in enumerate(["T-1", "T-2", "T-3", "T-4", "T-5"], 1)}


def helpdesk_world(admin: bool = False, nondet: bool = False, queue: bool = False) -> PyWorld:
    tables: dict[str, Any] = {"tickets": copy.deepcopy(TICKETS), "kb": copy.deepcopy(KB)}
    if queue:
        tables["queue"] = copy.deepcopy(QUEUE)
    desk = App("desk", tables, positional=["queue"] if queue else ())
    mail = App("mail", {"outbox": {}})

    @desk.tool(read_only=True)
    def get_ticket(ctx, ticket_id: str) -> dict:
        """One ticket: customer, subject, status (open or resolved), priority, resolution."""
        r = ctx.tables["tickets"].get(ticket_id)
        if r is None:
            raise ToolError(f"no ticket {ticket_id}")
        return dict(r)

    @desk.tool(read_only=True)
    def list_tickets(ctx, status: str = "open") -> dict:
        """The tickets with a status (open or resolved)."""
        return {"tickets": [dict(r) for r in ctx.tables["tickets"].values() if r["status"] == status]}

    @desk.tool(read_only=True)
    def search_kb(ctx, query: str) -> dict:
        """Knowledge base articles whose title or body contains the query: id and title."""
        q = query.casefold()
        return {"articles": [{"id": a["id"], "title": a["title"]} for a in ctx.tables["kb"].values() if q in a["title"].casefold() or q in a["body"].casefold()]}

    @desk.tool(read_only=True)
    def read_article(ctx, article_id: str) -> dict:
        """One knowledge base article."""
        a = ctx.tables["kb"].get(article_id)
        if a is None:
            raise ToolError(f"no article {article_id}")
        return dict(a)

    @desk.tool(writes=["tickets"])
    def resolve_ticket(ctx, ticket_id: str, resolution: str) -> dict:
        """Resolve an open ticket (status resolved) with a resolution: the id of the article that answers it."""
        t = ctx.tables["tickets"].get(ticket_id)
        if t is None or t["status"] != "open":
            raise ToolError(f"ticket {ticket_id} is not open")
        t["status"], t["resolution"] = "resolved", resolution
        if nondet:
            t["ref"] = uuid.uuid4().hex  # planted D11: run noise the contract reads
        return {"id": ticket_id, "status": "resolved", "resolution": resolution}

    @desk.tool(writes=["tickets"])
    def set_priority(ctx, ticket_id: str, priority: str) -> dict:
        """Set a ticket's priority: low, normal or high."""
        t = ctx.tables["tickets"].get(ticket_id)
        if t is None or priority not in {"low", "normal", "high"}:
            raise ToolError("bad ticket or priority")
        t["priority"] = priority
        return {"id": ticket_id, "priority": priority}

    if admin:
        @desk.tool(writes=["tickets"])
        def admin_update(ctx, ticket_id: str, status: str, resolution: str) -> dict:
            """Administrator override: write a ticket's status and resolution directly."""
            t = ctx.tables["tickets"][ticket_id]
            t["status"], t["resolution"] = status, resolution
            return {"id": ticket_id}

    if queue:
        @desk.tool(writes=["queue"])
        def claim(ctx, ticket_id: str) -> dict:
            """Claim the work-queue item of a ticket (state claimed)."""
            for row in ctx.tables["queue"].values():
                if row["ticket"] == ticket_id:
                    row["state"] = "claimed"
                    return {"ticket": ticket_id, "state": "claimed"}
            raise ToolError(f"no queue item for {ticket_id}")

    @mail.tool(writes=["outbox"])
    def send(ctx, to: str, subject: str, body: str) -> dict:
        """Email someone."""
        box = ctx.tables["outbox"]
        nid = max(box, default=0) + 1
        box[nid] = {"id": nid, "to": to, "subject": subject, "body": body}
        return {"id": nid, "sent": True}

    return PyWorld([desk, mail])


HD_N = Normalizers(id_canon=[IdCanon(app="mail", coll="outbox")])


def _w(factory: str, **params: Any) -> WorldRef:
    return WorldRef(factory=f"{MOD}:{factory}", params=params)


# ---------------------------------------------------------------------------------------------------------- correct tasks

def _hold(key: str = "103", reason: str = "price_variance", pid: str = "r") -> list[EffectPattern]:
    return [EffectPattern(id=f"{pid}_status", app="ap", coll="invoices", key=key, field="status", value=PEq(v="on_hold")),
            EffectPattern(id=f"{pid}_reason", app="ap", coll="invoices", key=key, field="reason", value=PEq(v=reason)),
            EffectPattern(id=f"{pid}_audit", app="ap", coll="audit_log", kind="insert", field="*",
                          where={"invoice_id": PEq(v=int(key)), "action": PEq(v="hold"), "reason": PEq(v=reason)})]


def t1_hold() -> Task:
    """Hold an over-tolerance invoice; the reason code is derived (from the tool catalogue), never stated."""
    return Task(id="ap/hold-1003", domain="synth-test", world=_w("ap_world"),
                instruction="Invoice INV-1003 bills more than its purchase order allows. Check it against the purchase order and put it on hold "
                            "with the policy's reason code.",
                oracle=Program(steps=[Step(id="s1", tool="ap.find_invoice", args={"number": lit("INV-1003")}),
                                      Step(id="s2", tool="ap.get_purchase_order", args={"po_id": ref("s1.invoices.0.po")}),
                                      Step(id="s3", tool="ap.hold_invoice", args={"invoice_id": ref("s1.invoices.0.id"), "reason": lit("price_variance")})]),
                contract=Contract(required=_hold(), normalizers=AP_N),
                intent=IntentCard(skill="3-way match: hold on price variance", derive=["price_variance"]))


def t2_approve_answer() -> Task:
    return Task(id="ap/approve-1001", domain="synth-test", world=_w("ap_world"),
                instruction="Approve invoice INV-1001 and report the total value of the goods receipts recorded against its purchase order.",
                oracle=Program(steps=[Step(id="s1", tool="ap.find_invoice", args={"number": lit("INV-1001")}),
                                      Step(id="s2", tool="ap.get_receipts", args={"po_id": ref("s1.invoices.0.po")}),
                                      Step(id="s3", tool="ap.approve_invoice", args={"invoice_id": ref("s1.invoices.0.id")})],
                               answer=ref("s2.total")),
                contract=Contract(required=[
                    EffectPattern(id="r_status", app="ap", coll="invoices", key="101", field="status", value=PEq(v="approved")),
                    EffectPattern(id="r_audit", app="ap", coll="audit_log", kind="insert", field="*",
                                  where={"invoice_id": PEq(v=101), "action": PEq(v="approve"), "reason": PEq(v=None)})],
                    answer=[AnswerCheck(id="q_total", op="approx", value=1200.0, tol=0.01)], normalizers=AP_N),
                intent=IntentCard(skill="approve and report receipts"))


def t3_resolve_email() -> Task:
    return Task(id="desk/resolve-T7", domain="synth-test", world=_w("helpdesk_world"),
                instruction="Ticket T-7 asks how to do a password reset. Find the knowledge base article on password reset, resolve the ticket with "
                            "the article id as the resolution, and email the customer the article title.",
                oracle=Program(steps=[Step(id="s1", tool="desk.get_ticket", args={"ticket_id": lit("T-7")}),
                                      Step(id="s2", tool="desk.search_kb", args={"query": lit("password reset")}),
                                      Step(id="s3", tool="desk.resolve_ticket", args={"ticket_id": lit("T-7"), "resolution": ref("s2.articles.0.id")}),
                                      Step(id="s4", tool="mail.send", args={"to": ref("s1.customer"), "subject": ref("s1.subject"),
                                                                             "body": ref("s2.articles.0.title")})]),
                contract=Contract(required=[
                    EffectPattern(id="r_status", app="desk", coll="tickets", key="T-7", field="status", value=PEq(v="resolved")),
                    EffectPattern(id="r_res", app="desk", coll="tickets", key="T-7", field="resolution", value=PEq(v="KB-3")),
                    EffectPattern(id="r_mail", app="mail", coll="outbox", kind="insert", field="*",
                                  where={"to": PEq(v="ann@example.com"), "subject": PEq(v="Password reset"), "body": PEq(v="How to do a password reset")})],
                    normalizers=HD_N),
                intent=IntentCard(skill="answer from the knowledge base", derive=["KB-3"]))


def t4_priority() -> Task:
    sel = fn("select", ref("s1.tickets"), lit({"customer": "bob@example.com"}))
    return Task(id="desk/priority-bob", domain="synth-test", world=_w("helpdesk_world"),
                instruction="Set the priority of every open ticket from bob@example.com to high.",
                oracle=Program(steps=[Step(id="s1", tool="desk.list_tickets", args={"status": lit("open")}),
                                      Step(id="s2", tool="desk.set_priority", args={"ticket_id": fn("get", sel, lit("0.id")), "priority": lit("high")}),
                                      Step(id="s3", tool="desk.set_priority", args={"ticket_id": fn("get", sel, lit("1.id")), "priority": lit("high")})]),
                contract=Contract(required=[
                    EffectPattern(id="r_t2", app="desk", coll="tickets", key="T-2", field="priority", value=PEq(v="high")),
                    EffectPattern(id="r_t5", app="desk", coll="tickets", key="T-5", field="priority", value=PEq(v="high"))], normalizers=HD_N),
                intent=IntentCard(skill="select by customer", derive=["T-2", "T-5"]))


def t5_batch() -> Task:
    """Three decisions in a World with 8 more pending invoices, so MUT reaches n >= 59 distinct effects (U_MUT <= 0.05 at
    c = 0); in the 8-invoice World the same task has only 27 (it would be admitted as provisional)."""
    s = [Step(id="s1", tool="ap.find_invoice", args={"number": lit("INV-1002")}),
         Step(id="s2", tool="ap.find_invoice", args={"number": lit("INV-1005")}),
         Step(id="s3", tool="ap.find_invoice", args={"number": lit("INV-1006")}),
         Step(id="s4", tool="ap.approve_invoice", args={"invoice_id": ref("s1.invoices.0.id")}),
         Step(id="s5", tool="ap.reject_invoice", args={"invoice_id": ref("s2.invoices.0.id"), "reason": lit("duplicate")}),
         Step(id="s6", tool="ap.hold_invoice", args={"invoice_id": ref("s3.invoices.0.id"), "reason": lit("missing_receipt")})]
    req = [EffectPattern(id="r102", app="ap", coll="invoices", key="102", field="status", value=PEq(v="approved")),
           EffectPattern(id="r105s", app="ap", coll="invoices", key="105", field="status", value=PEq(v="rejected")),
           EffectPattern(id="r105r", app="ap", coll="invoices", key="105", field="reason", value=PEq(v="duplicate")),
           EffectPattern(id="r106s", app="ap", coll="invoices", key="106", field="status", value=PEq(v="on_hold")),
           EffectPattern(id="r106r", app="ap", coll="invoices", key="106", field="reason", value=PEq(v="missing_receipt")),
           EffectPattern(id="a102", app="ap", coll="audit_log", kind="insert", field="*", where={"invoice_id": PEq(v=102), "action": PEq(v="approve"), "reason": PEq(v=None)}),
           EffectPattern(id="a105", app="ap", coll="audit_log", kind="insert", field="*",
                         where={"invoice_id": PEq(v=105), "action": PEq(v="reject"), "reason": PEq(v="duplicate")}),
           EffectPattern(id="a106", app="ap", coll="audit_log", kind="insert", field="*",
                         where={"invoice_id": PEq(v=106), "action": PEq(v="hold"), "reason": PEq(v="missing_receipt")})]
    return Task(id="ap/batch-borealis", domain="synth-test", world=_w("ap_world", extra=8),
                instruction="Three invoices from Borealis Freight are pending. Approve INV-1002, reject INV-1005 as a duplicate, and put INV-1006 on "
                            "hold for missing_receipt.",
                oracle=Program(steps=s), contract=Contract(required=req, normalizers=AP_N), intent=IntentCard(skill="apply stated decisions"))


CORRECT = {"t1": t1_hold, "t2": t2_approve_answer, "t3": t3_resolve_email, "t4": t4_priority, "t5": t5_batch}


def t6_resolve() -> Task:
    """Resolve only (the D8 and D11 fixtures change its World)."""
    return Task(id="desk/resolve-T3", domain="synth-test", world=_w("helpdesk_world"),
                instruction="Resolve ticket T-3 with the id of the knowledge base article on refund policy as the resolution.",
                oracle=Program(steps=[Step(id="s1", tool="desk.get_ticket", args={"ticket_id": lit("T-3")}),
                                      Step(id="s2", tool="desk.search_kb", args={"query": lit("refund policy")}),
                                      Step(id="s3", tool="desk.resolve_ticket", args={"ticket_id": lit("T-3"), "resolution": ref("s2.articles.0.id")})]),
                contract=Contract(required=[
                    EffectPattern(id="r_status", app="desk", coll="tickets", key="T-3", field="status", value=PEq(v="resolved")),
                    EffectPattern(id="r_res", app="desk", coll="tickets", key="T-3", field="resolution", value=PEq(v="KB-2"))], normalizers=HD_N),
                intent=IntentCard(skill="answer from the knowledge base", derive=["KB-2"]))


# ---------------------------------------------------------------------------------------------------------- planted defects

def _with(task: Task, **upd: Any) -> Task:
    return task.model_copy(update=upd, deep=True)


def planted(code: str) -> Task:
    """One task per defect class, made by the §8 operator named in the comment."""
    t1 = t1_hold()
    if code == "D1":  # drop_required: the email is no longer required (and is merely allowed)
        t = t3_resolve_email()
        c = t.contract.model_copy(update={"required": [p for p in t.contract.required if p.id != "r_mail"],
                                          "allowed": [EffectPattern(id="a_mail", app="mail", coll="outbox", kind="insert", field="*", why="customer emails are allowed")]})
        return _with(t, id="planted/D1", contract=c)
    if code == "D2":  # widen_frame: (ap, *, *, *) (both collections a decision writes)
        c = t1.contract.model_copy(update={"allowed": [EffectPattern(id="a_inv", app="ap", coll="invoices", kind="any", why="invoice edits are routine"),
                                                       EffectPattern(id="a_log", app="ap", coll="audit_log", kind="any", why="audit rows are routine")]})
        return _with(t1, id="planted/D2", contract=c)
    if code == "D3":  # relax_value: the answer tolerance x 50,000
        t = t2_approve_answer()
        c = t.contract.model_copy(update={"answer": [AnswerCheck(id="q_total", op="approx", value=1200.0, tol=500.0)]})
        return _with(t, id="planted/D3", contract=c)
    if code == "D4":  # vacuous_required: a requirement s0 already satisfies
        c = t1.contract.model_copy(update={"required": [*t1.contract.required,
                                                        EffectPattern(id="r_vac", app="ap", coll="invoices", key="104", field="status", kind="state", value=PEq(v="pending"))]})
        return _with(t1, id="planted/D4", contract=c)
    if code == "D5":  # hide_info: the sentence naming the invoice is gone
        return _with(t1, id="planted/D5", instruction="An invoice bills more than its purchase order allows. Check it against the purchase order and put it "
                                                      "on hold with the policy's reason code.")
    if code == "D6":  # over_constrain: an unstated order demand
        c = t1.contract.model_copy(update={"trace": [TraceConstraint(id="m_po_first", op="before", tool="ap.get_purchase_order", then="ap.hold_invoice")]})
        return _with(t1, id="planted/D6", contract=c)
    if code == "D7":  # leak_decision
        return _with(t1, id="planted/D7", instruction="Invoice INV-1003 bills more than its purchase order allows. Put it on hold because of a price_variance.")
    if code == "D8":  # plant_affordance: a direct-write admin tool
        return _with(t6_resolve(), id="planted/D8", world=_w("helpdesk_world", admin=True))
    if code == "D9":  # instruction_drift: any invoice will do
        req = [p.model_copy(update={"key": "*"}) for p in t1.contract.required]
        req[2] = req[2].model_copy(update={"where": {**req[2].where, "invoice_id": PAny()}})
        return _with(t1, id="planted/D9", contract=t1.contract.model_copy(update={"required": req}))
    if code == "D10":  # break_solvability: a reason code no tool can set
        req = [p if p.id != "r_reason" else p.model_copy(update={"value": PEq(v="fraud_suspected")}) for p in t1.contract.required]
        return _with(t1, id="planted/D10", contract=t1.contract.model_copy(update={"required": req}))
    if code == "D11":  # inject_nondet: resolve stamps a random ref the contract reads (allowed, but no normalizer drops it)
        t = _with(t6_resolve(), id="planted/D11", world=_w("helpdesk_world", nondet=True))
        allowed = [EffectPattern(id="a_ref", app="desk", coll="tickets", key="T-3", field="ref", why="the ticket system stamps a reference")]
        return _with(t, contract=t.contract.model_copy(update={"allowed": allowed}))
    raise KeyError(code)


def position_keyed() -> Task:
    """A contract keyed on a row position (the queue is canonicalized positionally): caught by PERT."""
    return Task(id="desk/claim-T4", domain="synth-test", world=_w("helpdesk_world", queue=True),
                instruction="Claim the work-queue item of ticket T-4.",
                oracle=Program(steps=[Step(id="s1", tool="desk.claim", args={"ticket_id": lit("T-4")})]),
                contract=Contract(required=[EffectPattern(id="r_q", app="desk", coll="queue", key="3", field="state", value=PEq(v="claimed"))],
                                  normalizers=HD_N),
                intent=IntentCard(skill="claim work"))


def always_pass() -> Task:
    return _with(t1_hold(), id="planted/always-pass", contract=Contract(allowed=[EffectPattern(id="all", kind="any", why="anything goes")]))


def in_domain(t: Task) -> Task:
    """The task in its World's domain (``ap-test`` or ``desk-test``): M3's cross-domain checks (birth gate, R7,
    natural-likeness) need two domains, and the M2 fixtures all share ``synth-test``."""
    return t.model_copy(update={"domain": "ap-test" if t.world.factory.endswith(":ap_world") else "desk-test"}, deep=True)


def anchors() -> list[Task]:
    """The certified anchors of M3 (clean certificates in M2): t1-t4 and t6 (t5's larger World is slower to plant)."""
    return [in_domain(f()) for f in (t1_hold, t2_approve_answer, t3_resolve_email, t4_priority, t6_resolve)]


__all__ = ["AP_N", "CORRECT", "HD_N", "PApprox", "always_pass", "anchors", "ap_world", "helpdesk_world", "in_domain", "planted", "position_keyed",
           "t6_resolve"]
