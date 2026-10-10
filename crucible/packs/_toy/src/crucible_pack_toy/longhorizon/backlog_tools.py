"""Backlog ticketing tools (copied into the forged pack as ``helpdesk/tools/backlog.py``). Plain functions over SQLite;
``datetime.now()`` is the World's virtual clock (``world.packaging.virtual_env``), so refund times are episode times."""
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

_DB = Path(__file__).resolve().parent.parent / "db" / "tickets.db"
_STATUSES = {"open", "pending", "resolved", "escalated"}


def _con():
    con = sqlite3.connect(_DB)
    con.row_factory = sqlite3.Row
    return con


def list_tickets(status: str = "open", offset: int = 0, limit: int = 25) -> dict:
    """List tickets with a status, a page at a time (offset/limit, at most 50)."""
    if status not in _STATUSES:
        raise ValueError(f"status must be one of {sorted(_STATUSES)}")
    limit = max(1, min(int(limit), 50))
    with _con() as c:
        total = c.execute("SELECT COUNT(*) FROM tickets WHERE status = ?", (status,)).fetchone()[0]
        rows = [dict(r) for r in c.execute("SELECT id, customer_id, subject, status FROM tickets WHERE status = ? ORDER BY id LIMIT ? OFFSET ?",
                                           (status, limit, int(offset)))]
    return {"total": total, "offset": int(offset), "tickets": rows}


def get_ticket(ticket_id: int) -> dict:
    """One ticket with its customer, the customer's orders, the conversation thread and the notes."""
    with _con() as c:
        t = c.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if t is None:
            raise ValueError(f"no ticket {ticket_id}")
        cust = dict(c.execute("SELECT * FROM customers WHERE id = ?", (t["customer_id"],)).fetchone())
        orders = [dict(r) for r in c.execute("SELECT id, amount_usd, status FROM orders WHERE customer_id = ? ORDER BY id", (t["customer_id"],))]
        notes = [dict(r) for r in c.execute("SELECT author, body FROM notes WHERE ticket_id = ? ORDER BY id", (ticket_id,))]
    d = dict(t)
    d["thread"] = json.loads(d.pop("thread") or "[]")
    return {"ticket": d, "customer": cust, "orders": orders, "notes": notes}


def get_order(order_id: str) -> dict:
    """One order with every refund already issued on it."""
    with _con() as c:
        o = c.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        if o is None:
            raise ValueError(f"no order {order_id}")
        refunds = [dict(r) for r in c.execute("SELECT ticket_id, amount_usd, approved_by, created_at FROM refunds WHERE order_id = ? ORDER BY id", (order_id,))]
    return {"order": dict(o), "refunds": refunds}


def add_note(ticket_id: int, body: str) -> dict:
    """Add an internal note to a ticket."""
    with _con() as c:
        if c.execute("SELECT 1 FROM tickets WHERE id = ?", (ticket_id,)).fetchone() is None:
            raise ValueError(f"no ticket {ticket_id}")
        c.execute("INSERT INTO notes(ticket_id, author, body) VALUES (?, 'agent', ?)", (ticket_id, body))
    return {"ok": True}


def set_status(ticket_id: int, status: str) -> dict:
    """Set a ticket's status (open, pending, resolved, escalated)."""
    if status not in _STATUSES:
        raise ValueError(f"status must be one of {sorted(_STATUSES)}")
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    with _con() as c:
        n = c.execute("UPDATE tickets SET status = ?, resolved_at = CASE WHEN ? = 'resolved' THEN ? ELSE resolved_at END WHERE id = ?",
                      (status, status, now, ticket_id)).rowcount
    if not n:
        raise ValueError(f"no ticket {ticket_id}")
    return {"id": ticket_id, "status": status}


def issue_refund(ticket_id: int, order_id: str, amount_usd: float, approved_by: str = "") -> dict:
    """Refund (part of) an order linked to a ticket. Record approved_by when the policy requires a supervisor's approval."""
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    with _con() as c:
        t = c.execute("SELECT customer_id FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        o = c.execute("SELECT customer_id, amount_usd FROM orders WHERE id = ?", (order_id,)).fetchone()
        if t is None or o is None or t["customer_id"] != o["customer_id"]:
            raise ValueError("ticket and order do not belong to the same customer")
        if not 0 < amount_usd <= o["amount_usd"]:
            raise ValueError("refund amount must be positive and at most the order amount")
        c.execute("INSERT INTO refunds(ticket_id, order_id, amount_usd, approved_by, created_at) VALUES (?,?,?,?,?)",
                  (ticket_id, order_id, amount_usd, approved_by or None, now))
    return {"refunded": amount_usd, "order_id": order_id, "approved_by": approved_by or None, "at": now}
