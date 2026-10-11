"""Warehouse inventory tools: the toy pack's second App, fused with the helpdesk by the m13 G-9 ``fuse`` operator
(MiMo tools/<app>.py style: plain functions; DB next to this folder)."""
import sqlite3
from pathlib import Path

_DB = Path(__file__).resolve().parent.parent / "db" / "inventory.db"
_CONDITIONS = {"resellable", "damaged"}


def _con():
    con = sqlite3.connect(_DB)
    con.row_factory = sqlite3.Row
    return con


def get_order_items(order_id: str) -> list:
    """The items (sku, qty) shipped in an order."""
    with _con() as c:
        rows = [dict(r) for r in c.execute("SELECT sku, qty FROM order_items WHERE order_id = ? ORDER BY sku", (order_id,))]
    if not rows:
        raise ValueError(f"no shipped items for order {order_id}")
    return rows


def get_stock(sku: str) -> dict:
    """Units of a sku on the shelf."""
    with _con() as c:
        r = c.execute("SELECT sku, qty FROM stock WHERE sku = ?", (sku,)).fetchone()
    if r is None:
        raise ValueError(f"unknown sku {sku}")
    return dict(r)


def log_return(order_id: str, sku: str, qty: int, condition: str) -> dict:
    """Log a returned item. A resellable item goes back on the shelf; a damaged one goes to quarantine (not restocked)."""
    if condition not in _CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(_CONDITIONS)}")
    with _con() as c:
        shipped = c.execute("SELECT qty FROM order_items WHERE order_id = ? AND sku = ?", (order_id, sku)).fetchone()
        if shipped is None:
            raise ValueError(f"order {order_id} did not ship {sku}")
        if int(qty) < 1 or int(qty) > shipped["qty"]:
            raise ValueError(f"qty must be 1..{shipped['qty']}")
        cur = c.execute("INSERT INTO returns(order_id, sku, qty, condition) VALUES (?, ?, ?, ?)", (order_id, sku, int(qty), condition))
        if condition == "resellable":
            c.execute("UPDATE stock SET qty = qty + ? WHERE sku = ?", (int(qty), sku))
    return {"return_id": cur.lastrowid, "restocked": condition == "resellable"}


def list_returns() -> list:
    """Every logged return."""
    with _con() as c:
        return [dict(r) for r in c.execute("SELECT id, order_id, sku, qty, condition FROM returns ORDER BY id")]
