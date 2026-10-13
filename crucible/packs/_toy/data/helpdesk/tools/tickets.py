"""Helpdesk ticketing tools (MiMo tools/<app>.py style: plain functions; DB next to this folder)."""
import sqlite3
from pathlib import Path

_DB = Path(__file__).resolve().parent.parent / "db" / "tickets.db"
_STATUSES = {"open", "pending", "resolved", "escalated"}


def _con():
    con = sqlite3.connect(_DB)
    con.row_factory = sqlite3.Row
    return con


def _int(x, name: str) -> int:
    """An integer argument the database can bind (M14 audit V1: 10**20 raised OverflowError inside sqlite3)."""
    if isinstance(x, bool) or not isinstance(x, int) or not -2**63 <= x < 2**63:
        raise ValueError(f"{name} must be an integer between -2**63 and 2**63-1")
    return x


def list_tickets(status: str = "open") -> list:
    """List tickets with the given status."""
    if status not in _STATUSES:
        raise ValueError(f"status must be one of {sorted(_STATUSES)}")
    with _con() as c:
        return [dict(r) for r in c.execute("SELECT id, customer_id, subject, status, opened_at FROM tickets WHERE status = ? ORDER BY id", (status,))]


def get_ticket(ticket_id: int) -> dict:
    """Get one ticket with its customer, the customer's orders and the ticket notes."""
    ticket_id = _int(ticket_id, "ticket_id")
    with _con() as c:
        t = c.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if t is None:
            raise ValueError(f"no ticket {ticket_id}")
        cust = dict(c.execute("SELECT * FROM customers WHERE id = ?", (t["customer_id"],)).fetchone())
        orders = [dict(r) for r in c.execute("SELECT * FROM orders WHERE customer_id = ? ORDER BY id", (t["customer_id"],))]
        notes = [dict(r) for r in c.execute("SELECT author, body FROM notes WHERE ticket_id = ? ORDER BY id", (ticket_id,))]
    return {"ticket": dict(t), "customer": cust, "orders": orders, "notes": notes}


def add_note(ticket_id: int, body: str) -> dict:
    """Add an internal note to a ticket."""
    ticket_id = _int(ticket_id, "ticket_id")
    with _con() as c:
        if c.execute("SELECT 1 FROM tickets WHERE id = ?", (ticket_id,)).fetchone() is None:
            raise ValueError(f"no ticket {ticket_id}")
        c.execute("INSERT INTO notes(ticket_id, author, body) VALUES (?, 'agent', ?)", (ticket_id, body))
    return {"ok": True}


def set_status(ticket_id: int, status: str, now: str = "2026-03-02T09:00:00Z") -> dict:
    """Set a ticket's status (open, pending, resolved, escalated)."""
    ticket_id = _int(ticket_id, "ticket_id")
    if status not in _STATUSES:
        raise ValueError(f"status must be one of {sorted(_STATUSES)}")
    with _con() as c:
        n = c.execute("UPDATE tickets SET status = ?, resolved_at = CASE WHEN ? = 'resolved' THEN ? ELSE resolved_at END WHERE id = ?",
                      (status, status, now, ticket_id)).rowcount
    if not n:
        raise ValueError(f"no ticket {ticket_id}")
    return {"id": ticket_id, "status": status}


def request_approval(ticket_id: int, order_id: str, amount_usd: float, reason: str = "") -> dict:
    """Ask the supervisor to approve a refund (the approval workflow). Returns the decision and, when approved, the
    approver's name to record in issue_refund's approved_by. Orders in transit are never approved."""
    ticket_id = _int(ticket_id, "ticket_id")
    with _con() as c:
        t = c.execute("SELECT customer_id FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        o = c.execute("SELECT customer_id, amount_usd, status FROM orders WHERE id = ?", (order_id,)).fetchone()
        if t is None or o is None or t["customer_id"] != o["customer_id"]:
            raise ValueError("ticket and order do not belong to the same customer")
        if not 0 < amount_usd <= o["amount_usd"]:
            raise ValueError("refund amount must be positive and at most the order amount")
        if o["status"] == "in_transit":
            status, approver, why = "rejected", None, "orders in transit are not refundable; escalate for a carrier trace"
        else:
            sup = c.execute("SELECT name FROM staff WHERE role = 'refund_supervisor'").fetchone()
            if sup is None:
                raise RuntimeError("no refund supervisor configured in this world")  # a world defect: infra, not the policy's
            status, approver, why = "approved", sup["name"], reason or None
        c.execute("INSERT INTO approvals(ticket_id, order_id, amount_usd, status, approver, reason) VALUES (?,?,?,?,?,?)",
                  (ticket_id, order_id, amount_usd, status, approver, why))
    return {"status": status, "approver": approver, "ticket_id": ticket_id, "order_id": order_id, "amount_usd": amount_usd, "reason": why}


def issue_refund(ticket_id: int, order_id: str, amount_usd: float, approved_by: str = "", now: str = "2026-03-02T09:00:00Z") -> dict:
    """Refund (part of) an order linked to a ticket. Refunds above 50 USD need approved_by (a supervisor)."""
    ticket_id = _int(ticket_id, "ticket_id")
    with _con() as c:
        t = c.execute("SELECT customer_id FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        o = c.execute("SELECT customer_id, amount_usd FROM orders WHERE id = ?", (order_id,)).fetchone()
        if t is None or o is None or t["customer_id"] != o["customer_id"]:
            raise ValueError("ticket and order do not belong to the same customer")
        if not 0 < amount_usd <= o["amount_usd"]:
            raise ValueError("refund amount must be positive and at most the order amount")
        c.execute("INSERT INTO refunds(ticket_id, order_id, amount_usd, approved_by, created_at) VALUES (?,?,?,?,?)",
                  (ticket_id, order_id, amount_usd, approved_by or None, now))
    return {"refunded": amount_usd, "order_id": order_id, "approved_by": approved_by or None}
