"""Accounts-payable ERP tools (copied into each generated world as ``ap/tools/erp.py``). Plain functions over SQLite."""
import sqlite3
from pathlib import Path

_DB = Path(__file__).resolve().parent.parent / "db" / "erp.db"
HOLD_REASONS = {"price_variance", "missing_receipt", "bank_verification"}
REJECT_REASONS = {"duplicate", "vendor_blocked"}


def _con():
    con = sqlite3.connect(_DB)
    con.row_factory = sqlite3.Row
    return con


def list_invoices(status: str = "pending", offset: int = 0, limit: int = 25) -> dict:
    """List invoices with a status (pending, approved, on_hold, rejected), a page at a time."""
    if status not in {"pending", "approved", "on_hold", "rejected"}:
        raise ValueError("status must be pending, approved, on_hold or rejected")
    limit = max(1, min(int(limit), 50))
    with _con() as c:
        total = c.execute("SELECT COUNT(*) FROM invoices WHERE status = ?", (status,)).fetchone()[0]
        rows = [dict(r) for r in c.execute("SELECT id, vendor_id, invoice_number, amount, status FROM invoices WHERE status = ? ORDER BY id LIMIT ? OFFSET ?",
                                           (status, limit, int(offset)))]
    return {"total": total, "offset": int(offset), "invoices": rows}


def get_invoice(invoice_id: int) -> dict:
    """One invoice: vendor, invoice number, the purchase order it bills, amount, status."""
    with _con() as c:
        r = c.execute("SELECT * FROM invoices WHERE id = ?", (invoice_id,)).fetchone()
    if r is None:
        raise ValueError(f"no invoice {invoice_id}")
    return dict(r)


def get_purchase_order(po_id: str) -> dict:
    """A purchase order: vendor and approved amount."""
    with _con() as c:
        r = c.execute("SELECT * FROM purchase_orders WHERE id = ?", (po_id,)).fetchone()
    if r is None:
        raise ValueError(f"no purchase order {po_id}")
    return dict(r)


def get_receipts(po_id: str) -> dict:
    """Goods receipts recorded against a purchase order, and their total value."""
    with _con() as c:
        rows = [dict(r) for r in c.execute("SELECT id, po_id, value, received_on FROM receipts WHERE po_id = ? ORDER BY id", (po_id,))]
    return {"po_id": po_id, "receipts": rows, "total_received": round(sum(r["value"] for r in rows), 2)}


def get_vendor(vendor_id: str) -> dict:
    """A vendor: status (active or blocked) and whether its bank details changed without verification."""
    with _con() as c:
        r = c.execute("SELECT * FROM vendors WHERE id = ?", (vendor_id,)).fetchone()
    if r is None:
        raise ValueError(f"no vendor {vendor_id}")
    return dict(r)


def find_invoices(vendor_id: str, invoice_number: str) -> dict:
    """Every invoice (any status) a vendor sent with this invoice number."""
    with _con() as c:
        rows = [dict(r) for r in c.execute("SELECT id, invoice_number, amount, status FROM invoices WHERE vendor_id = ? AND invoice_number = ? ORDER BY id",
                                           (vendor_id, invoice_number))]
    return {"invoices": rows}


def _decide(invoice_id: int, status: str, reason: str | None) -> dict:
    with _con() as c:
        r = c.execute("SELECT status FROM invoices WHERE id = ?", (invoice_id,)).fetchone()
        if r is None:
            raise ValueError(f"no invoice {invoice_id}")
        if r["status"] != "pending":
            raise ValueError(f"invoice {invoice_id} is {r['status']}, not pending")
        c.execute("UPDATE invoices SET status = ?, reason = ? WHERE id = ?", (status, reason, invoice_id))
    return {"id": invoice_id, "status": status, "reason": reason}


def approve_invoice(invoice_id: int) -> dict:
    """Approve a pending invoice for payment."""
    return _decide(invoice_id, "approved", None)


def hold_invoice(invoice_id: int, reason: str) -> dict:
    """Put a pending invoice on hold. reason: price_variance, missing_receipt or bank_verification."""
    if reason not in HOLD_REASONS:
        raise ValueError(f"reason must be one of {sorted(HOLD_REASONS)}")
    return _decide(invoice_id, "on_hold", reason)


def reject_invoice(invoice_id: int, reason: str) -> dict:
    """Reject a pending invoice. reason: duplicate or vendor_blocked."""
    if reason not in REJECT_REASONS:
        raise ValueError(f"reason must be one of {sorted(REJECT_REASONS)}")
    return _decide(invoice_id, "rejected", reason)
