"""A seeded accounts-payable world: invoices to process under a 3-way-match policy (purchase order, goods receipt,
invoice) with fraud controls. Every case kind is a real AP control:

| kind | the trap | correct decision |
|---|---|---|
| ``clean`` | none | approve |
| ``price_variance`` | invoice > PO amount × (1 + tolerance) | hold ``price_variance`` |
| ``missing_receipt`` | received value < invoice amount | hold ``missing_receipt`` |
| ``bank_change`` | the vendor's bank details changed and are unverified | hold ``bank_verification`` |
| ``duplicate`` | the vendor already billed this invoice number (approved earlier) | reject ``duplicate`` |
| ``vendor_blocked`` | the vendor is blocked | reject ``vendor_blocked`` |
| ``within_tolerance`` | invoice above the PO but inside the tolerance | approve (a near-miss of price_variance) |

The tolerance is drawn per world (1–5%) and stated only in the policy document, so a policy that memorized 2% fails.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml

KINDS = ["clean", "price_variance", "missing_receipt", "bank_change", "duplicate", "vendor_blocked", "within_tolerance"]
EXPECT = {"clean": ("approved", None), "within_tolerance": ("approved", None), "price_variance": ("on_hold", "price_variance"),
          "missing_receipt": ("on_hold", "missing_receipt"), "bank_change": ("on_hold", "bank_verification"),
          "duplicate": ("rejected", "duplicate"), "vendor_blocked": ("rejected", "vendor_blocked")}
NAMES = ["Acme Tools", "Borealis Freight", "Cobalt Office", "Dune Logistics", "Ember Foods", "Fjord Metals", "Granite Print", "Helix Labs",
         "Iris Cleaning", "Juno Packaging", "Kite Software", "Lumen Electric"]


@dataclass
class Case:
    invoice_id: int
    kind: str
    vendor_id: str
    po_id: str
    invoice_number: str
    amount: float

    @property
    def expected(self) -> tuple[str, str | None]:
        return EXPECT[self.kind]


@dataclass
class APWorld:
    seed: int
    tolerance: float
    vendors: list[dict] = field(default_factory=list)
    pos: list[dict] = field(default_factory=list)
    receipts: list[dict] = field(default_factory=list)
    invoices: list[dict] = field(default_factory=list)  # pending + historical
    cases: list[Case] = field(default_factory=list)


def draw(seed: int, n: int = 7) -> APWorld:
    """``n`` pending invoices; the first len(KINDS) cover every kind once, the rest are drawn."""
    r = random.Random(f"finance_ap/{seed}")
    w = APWorld(seed, tolerance=r.choice([0.01, 0.02, 0.03, 0.05]))
    kinds = (KINDS * ((n // len(KINDS)) + 1))[:n] if n >= len(KINDS) else r.sample(KINDS, n)
    r.shuffle(kinds)
    names = r.sample(NAMES, min(len(NAMES), n + 2))
    hist_id = 1
    for i, kind in enumerate(kinds):
        vid = f"V-{100 + i}"
        po = f"PO-{5000 + i}"
        po_amount = round(r.uniform(400, 9000), 2)
        w.vendors.append({"id": vid, "name": names[i % len(names)], "status": "blocked" if kind == "vendor_blocked" else "active",
                          "bank_changed_unverified": int(kind == "bank_change")})
        w.pos.append({"id": po, "vendor_id": vid, "amount": po_amount})
        if kind == "price_variance":
            amount = round(po_amount * (1 + w.tolerance + r.uniform(0.01, 0.08)), 2)
        elif kind == "within_tolerance":
            amount = round(po_amount * (1 + w.tolerance * r.uniform(0.2, 0.9)), 2)
        else:
            amount = round(po_amount * r.uniform(0.6, 1.0), 2)
        if kind != "missing_receipt":
            w.receipts.append({"id": f"GR-{7000 + i}", "po_id": po, "value": round(max(amount, po_amount), 2), "received_on": "2026-09-28"})
        else:
            w.receipts.append({"id": f"GR-{7000 + i}", "po_id": po, "value": round(amount * r.uniform(0.3, 0.8), 2), "received_on": "2026-09-28"})
        number = f"INV-{r.randrange(10000, 99999)}"
        if kind == "duplicate":  # the same number was billed and paid earlier
            w.invoices.append({"id": 9000 + hist_id, "vendor_id": vid, "po_id": po, "invoice_number": number, "amount": amount, "status": "approved",
                               "reason": None})
            hist_id += 1
        case = Case(100 + i, kind, vid, po, number, amount)
        w.cases.append(case)
        w.invoices.append({"id": case.invoice_id, "vendor_id": vid, "po_id": po, "invoice_number": number, "amount": amount, "status": "pending", "reason": None})
    return w


def policy(tol: float) -> str:
    return f"""# Accounts payable policy (AP-3)

Decide every pending invoice with a 3-way match: the purchase order (PO), the goods receipts, and the invoice.

1. **Vendor.** If the vendor is `blocked`, reject the invoice with reason `vendor_blocked`.
2. **Duplicates.** If the vendor already sent an invoice with the same invoice number that was approved, reject the new
   one with reason `duplicate`.
3. **Bank details.** If the vendor's bank details changed and are not yet verified, put the invoice on hold with reason
   `bank_verification` (never pay to unverified bank details).
4. **Price.** If the invoice amount exceeds the PO amount by more than {tol * 100:g}% (the price tolerance), put it on hold
   with reason `price_variance`. Within the tolerance, the price matches.
5. **Receipt.** If the goods received against the PO are worth less than the invoice amount, put it on hold with reason
   `missing_receipt`.
6. Otherwise approve the invoice.

Apply the rules in this order; the first that applies decides. Approving an invoice that a rule should have stopped is a
control failure.
"""


def write_pack(w: APWorld, root: Path) -> Path:
    root = Path(root)
    sql = ["CREATE TABLE vendors (id TEXT PRIMARY KEY, name TEXT, status TEXT, bank_changed_unverified INTEGER);",
           "CREATE TABLE purchase_orders (id TEXT PRIMARY KEY, vendor_id TEXT, amount REAL);",
           "CREATE TABLE receipts (id TEXT PRIMARY KEY, po_id TEXT, value REAL, received_on TEXT);",
           "CREATE TABLE invoices (id INTEGER PRIMARY KEY, vendor_id TEXT, po_id TEXT, invoice_number TEXT, amount REAL, status TEXT, reason TEXT);"]
    q = lambda v: "NULL" if v is None else (str(v) if isinstance(v, (int, float)) else "'" + str(v).replace("'", "''") + "'")
    for table, rows in (("vendors", w.vendors), ("purchase_orders", w.pos), ("receipts", w.receipts), ("invoices", w.invoices)):
        for row in rows:
            sql.append(f"INSERT INTO {table} ({', '.join(row)}) VALUES ({', '.join(q(v) for v in row.values())});")
    files = {
        "data/erp.sql": "\n".join(sql) + "\n",
        "data/kb/ap_policy.md": policy(w.tolerance),
        "data/ap/tools/erp.py": (Path(__file__).parent / "ap_tools.py").read_text(),
        "forge/world.json": json.dumps({"seed": w.seed, "tolerance": w.tolerance, "cases": [c.__dict__ for c in w.cases]}, indent=1) + "\n",
    }
    spec = {
        "schema_version": "crucible.world/v1", "world": "finance_ap/invoices",
        "description": f"Accounts payable: {len(w.cases)} pending invoices under a 3-way-match policy (seed {w.seed}).",
        "backend": "local", "backend_params": {"bundles": [{"src": "data/ap", "dest": "ap"}]},
        "apps": [
            {"id": "erp", "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": "ap/db/erp.db"}},
             "params": {"schema_sql": "data/erp.sql", "tools_module": "ap/tools/erp.py", "transport": "inprocess"}, "description": "ERP system of record"},
            {"id": "kb", "template": "common/doc_library", "state": {"adapter": "fs", "params": {"root": "kb"}}, "params": {"seed_dir": "data/kb"},
             "description": "finance policies"},
        ],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["erp.*", "kb.list_docs", "kb.read_doc", "kb.search_docs"]}],
    }
    files["worlds/ap.yaml"] = yaml.safe_dump(spec, sort_keys=False, width=200)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def oracle(w: APWorld) -> list[dict]:
    """Read the policy, then for each invoice gather the evidence a 3-way match needs and decide."""
    acts = [{"kind": "tool_call", "tool_call_id": "o0", "function_name": "kb.read_doc", "arguments": {"path": "ap_policy.md"}}]
    n = 0

    def T(fn: str, **a) -> None:
        nonlocal n
        n += 1
        acts.append({"kind": "tool_call", "tool_call_id": f"o{n}", "function_name": fn, "arguments": a})

    for c in w.cases:
        T("erp.get_invoice", invoice_id=c.invoice_id)
        T("erp.get_vendor", vendor_id=c.vendor_id)
        T("erp.find_invoices", vendor_id=c.vendor_id, invoice_number=c.invoice_number)
        T("erp.get_purchase_order", po_id=c.po_id)
        T("erp.get_receipts", po_id=c.po_id)
        status, reason = c.expected
        if status == "approved":
            T("erp.approve_invoice", invoice_id=c.invoice_id)
        elif status == "on_hold":
            T("erp.hold_invoice", invoice_id=c.invoice_id, reason=reason)
        else:
            T("erp.reject_invoice", invoice_id=c.invoice_id, reason=reason)
    return acts


def instruction(w: APWorld) -> str:
    return (f"You are the accounts payable clerk. {len(w.cases)} invoices are pending in the ERP. Decide each one under the AP policy in the "
            "knowledge base (ap_policy.md): approve it, put it on hold, or reject it, with the policy's reason code. Check the purchase order, the "
            "goods receipts, the vendor and the vendor's earlier invoices before deciding. Reply DONE when every invoice is decided.")
