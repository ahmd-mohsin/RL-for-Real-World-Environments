"""Vendor portal (toy website): invoices a clerk reviews and disputes through a browser.

Plain server-rendered HTML with labelled form controls, so an accessibility snapshot exposes them.
State lives in SQLite (the system of record verifiers read).
"""

import html
import sqlite3

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.routing import Route

SCHEMA = """
CREATE TABLE IF NOT EXISTS invoices (id TEXT PRIMARY KEY, vendor TEXT NOT NULL, amount_usd REAL NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','disputed')), note TEXT);
INSERT OR IGNORE INTO invoices(id, vendor, amount_usd) VALUES
  ('INV-1001', 'Acme Parts', 1200.50), ('INV-1002', 'Globex', 89.99), ('INV-1003', 'Acme Parts', 1200.50);
"""


def create_app(db_path: str) -> Starlette:
    con0 = sqlite3.connect(db_path)
    con0.executescript(SCHEMA)
    con0.commit()
    con0.close()

    def con():
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        return c

    def page(title: str, body: str) -> HTMLResponse:
        return HTMLResponse(f"<!doctype html><html><head><title>{html.escape(title)}</title></head><body><h1>{html.escape(title)}</h1>{body}</body></html>")

    async def index(_r: Request):
        with con() as c:
            rows = c.execute("SELECT id, vendor, amount_usd, status FROM invoices ORDER BY id").fetchall()
        items = "".join(f'<tr><td><a href="/invoices/{r["id"]}">{r["id"]}</a></td><td>{html.escape(r["vendor"])}</td>'
                        f'<td>{r["amount_usd"]:.2f}</td><td>{r["status"]}</td></tr>' for r in rows)
        return page("Vendor invoices", f"<table><tr><th>Invoice</th><th>Vendor</th><th>Amount (USD)</th><th>Status</th></tr>{items}</table>")

    async def detail(r: Request):
        inv = r.path_params["inv"]
        with con() as c:
            row = c.execute("SELECT * FROM invoices WHERE id = ?", (inv,)).fetchone()
        if row is None:
            return page("Not found", "<p>No such invoice.</p>")
        if r.method == "POST":
            form = await r.form()
            action, note = form.get("action"), (form.get("note") or "").strip()
            if action not in ("approve", "dispute"):
                return page("Error", "<p>Choose an action.</p>")
            if action == "dispute" and not note:
                return page("Error", "<p>A dispute needs a note.</p>")
            with con() as c:
                c.execute("UPDATE invoices SET status = ?, note = ? WHERE id = ?", ("approved" if action == "approve" else "disputed", note or None, inv))
            return RedirectResponse(f"/invoices/{inv}?saved=1", status_code=303)
        saved = '<p role="status">Saved.</p>' if r.query_params.get("saved") else ""
        form = ('<form method="post"><label for="action">Action</label><select id="action" name="action">'
                '<option value="">--</option><option value="approve">Approve</option><option value="dispute">Dispute</option></select>'
                '<label for="note">Note</label><textarea id="note" name="note"></textarea><button type="submit">Submit</button></form>')
        return page(f"Invoice {inv}", f'{saved}<p>Vendor: {html.escape(row["vendor"])}</p><p>Amount: {row["amount_usd"]:.2f} USD</p>'
                                      f'<p>Status: {row["status"]}</p><p><a href="/">All invoices</a></p>{form}')

    return Starlette(routes=[Route("/", index), Route("/invoices/{inv}", detail, methods=["GET", "POST"])])
