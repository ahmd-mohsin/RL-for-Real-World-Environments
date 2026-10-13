"""A seeded accounts-payable world, generator v3 (M14 audit D8, V6-V8, V10; campaign B7): pending invoices to decide under
the AP-3 policy, a 3-way match (purchase order, goods receipts, invoice) with fraud controls.

The label of every invoice is the AP-3 policy applied to the generated ERP data (``decide``), never a case kind. The
policy has seven rules, applied in order, and the first that applies decides:

| rule | the control | decision |
|---|---|---|
| 1 | the vendor is blocked, or is the same payee as a blocked vendor (another vendor record with the same bank account) | reject ``vendor_blocked`` |
| 2 | the vendor's earlier invoice with the same number (after normalization) was approved | reject ``duplicate`` |
| 3 | the purchase order does not exist or belongs to another vendor (the match is impossible) | hold ``po_mismatch`` |
| 4-6 | unverified bank change, price over tolerance, receipts short, **in this world's order** | hold with that reason |
| 7 | none applies | approve |

What a world draws (v1 drew one invoice of each of 7 kinds, so the composition, the rule order and a memorized 2%
tolerance were all learnable without reading anything):
- ``n`` (7-14 pending invoices) and the base rates: 45-70% of the drawn invoices trip no rule; the rest trip one rule,
  or two (a compound case, where only the rule order gives the answer).
- Per-world parameters stated only in the policy document: the price tolerance (0.5-10%, inclusive at the bar) and
  the order of the three hold rules. No constant tolerance and no fixed order is right in every world.
- Near misses: an amount exactly at the tolerance bar or one cent over it, receipts in parts that add up exactly,
  an earlier same-number invoice that was rejected or put on hold, the same number from another vendor, a number
  one digit off.
- Duplicates in the forms real duplicate payments take (case, spaces, punctuation, leading zeros, a restated amount).
  Every world holds at least one that an exact string match misses.
- An infeasible case: an invoice against a purchase order that does not exist or belongs to another vendor.

Every world holds (campaign B7: over 2,000 v2 worlds, ignoring the hold order solved 57.8%, whole-unit prices 68%,
amount-matched duplicates 54.5%, and no world held a vendor alias):
- two compound hold cases on the two **adjacent** pairs of the world's hold order, so every fixed order other than the
  world's own decides one of them wrongly (a fixed order is right only in the 1/6 of worlds that drew it);
- a duplicate whose number is a variant (an exact match misses it) and whose amount is restated (whole units, or a
  small correction), so a policy that needs the amount to match misses it;
- a price exactly one cent over the tolerance bar whose whole-currency-unit comparison is not over the bar;
- a payee the policy stops: in 85% of worlds a **vendor alias** (a blocked vendor re-registered under a new vendor id
  and a new name with the same bank account, found through the vendor master: ``list_vendors``), otherwise a blocked
  vendor's own invoice.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

GENERATOR_VERSION = 3
ALIAS_SHARE = 0.85  # worlds whose stopped payee is a vendor alias
ALIAS_SUFFIXES = ["Services", "Group", "Holdings", "Trading", "International", "Solutions"]
TOLERANCES_PM = [5, 10, 20, 30, 50, 75, 100]  # per mille: 0.5% .. 10%
HOLD_RULES = ["bank_verification", "price_variance", "missing_receipt"]
TRAPS = ["vendor_blocked", "duplicate", "po_mismatch", "bank_verification", "price_variance", "missing_receipt"]
NEAR_MISSES = ["within_tolerance", "near_tolerance", "at_tolerance", "receipts_exact", "dup_earlier_rejected", "dup_other_vendor", "dup_similar_number"]
HOLD_REASONS = {"price_variance", "missing_receipt", "bank_verification", "po_mismatch"}
REJECT_REASONS = {"duplicate", "vendor_blocked"}
NAMES = ["Acme Tools", "Borealis Freight", "Cobalt Office", "Dune Logistics", "Ember Foods", "Fjord Metals", "Granite Print", "Helix Labs",
         "Iris Cleaning", "Juno Packaging", "Kite Software", "Lumen Electric", "Maple Catering", "Nimbus Hosting", "Onyx Security",
         "Pioneer Fuels", "Quartz Glass", "Raven Couriers", "Sable Textiles", "Tundra Cold Chain", "Umber Paints", "Vertex Staffing",
         "Willow Furniture", "Xenon Lighting", "Yarrow Seeds", "Zephyr Air", "Atlas Scaffolding", "Beacon Signage", "Cedar Timber",
         "Delta Plumbing", "Echo Acoustics", "Flint Machining", "Garnet Tiles", "Harbor Marine", "Indigo Dyes", "Jasper Stone"]
_KIND = {"vendor_blocked": "vendor_blocked", "duplicate": "duplicate", "po_mismatch": "po_mismatch", "bank_verification": "bank_change",
         "price_variance": "price_variance", "missing_receipt": "missing_receipt"}


def norm_number(s: str) -> str:
    """AP-3's invoice-number match: ignore letter case, spaces and punctuation, and leading zeros of a number."""
    t = re.sub(r"[^A-Z0-9]", "", str(s).upper())
    return re.sub(r"(?<![0-9])0+(?=[0-9])", "", t)


def cents(x: float) -> int:
    return round(float(x) * 100)


@dataclass
class Case:
    invoice_id: int
    kind: str  # the rule that decides it (``bank_change`` for bank_verification), or ``clean`` / ``within_tolerance``
    vendor_id: str
    po_id: str
    invoice_number: str
    amount: float
    traits: list[str] = field(default_factory=list)  # the traps and near misses the generator built in
    expected: tuple[str, str | None] = ("approved", None)  # AP-3 applied to the data (``decide``)
    fired: list[str] = field(default_factory=list)  # every rule that applies, in this world's order

    @property
    def compound(self) -> bool:
        return len(self.fired) >= 2


@dataclass
class APWorld:
    seed: int
    tolerance: float  # a fraction (0.075 = 7.5%)
    tol_pm: int = 20
    hold_order: list[str] = field(default_factory=lambda: list(HOLD_RULES))
    vendors: list[dict] = field(default_factory=list)
    pos: list[dict] = field(default_factory=list)
    receipts: list[dict] = field(default_factory=list)
    invoices: list[dict] = field(default_factory=list)  # pending + historical
    cases: list[Case] = field(default_factory=list)


# ------------------------------------------------------------------------------------------------ the policy, as code

def rules_that_apply(w: APWorld, inv: dict, *, tol_pm: int | None = None, normalize: bool = True, aliases: bool = True,
                     dup_same_amount: bool = False, whole_units: bool = False) -> dict[str, bool]:
    """Which AP-3 conditions hold for one pending invoice, from the ERP data alone. ``aliases=False`` (only the vendor's
    own status), ``dup_same_amount`` (a duplicate must also bill the same amount) and ``whole_units`` (prices compared in
    whole currency units) describe shortcut readings for the adversaries; the labels never use them."""
    tol_pm = w.tol_pm if tol_pm is None else tol_pm
    v = next((x for x in w.vendors if x["id"] == inv["vendor_id"]), None)
    po = next((x for x in w.pos if x["id"] == inv["po_id"]), None)
    key = norm_number if normalize else str
    amt = cents(inv["amount"])
    dup = any(h["id"] != inv["id"] and h["vendor_id"] == inv["vendor_id"] and h["status"] == "approved"
              and key(h["invoice_number"]) == key(inv["invoice_number"]) and (not dup_same_amount or cents(h["amount"]) == amt) for h in w.invoices)
    blocked = bool(v and v["status"] == "blocked")
    if v and aliases and not blocked:  # the same payee under another vendor id: the same bank account as a blocked vendor
        blocked = any(x["id"] != v["id"] and x["status"] == "blocked" and x.get("bank_account") == v.get("bank_account") for x in w.vendors)
    if po is None:
        over = False
    elif whole_units:
        over = round(inv["amount"]) > round(po["amount"] * (1 + tol_pm / 1000))
    else:
        over = amt * 1000 > cents(po["amount"]) * (1000 + tol_pm)  # "more than tol%", exact in cents
    return {"vendor_blocked": blocked, "duplicate": dup,
            "po_mismatch": po is None or po["vendor_id"] != inv["vendor_id"],
            "bank_verification": bool(v and v["bank_changed_unverified"]),
            "price_variance": over,
            "missing_receipt": po is not None and sum(cents(r["value"]) for r in w.receipts if r["po_id"] == po["id"]) < amt}


def neighbour_tolerance(tol_pm: int) -> int:
    """Another company's tolerance: the next one in TOLERANCES_PM (the previous one for the largest). The world's two
    price probes (one at the bar or in the top fifth of the band, one a cent over the bar) make it wrong in every world."""
    k = TOLERANCES_PM.index(tol_pm)
    return TOLERANCES_PM[k + 1] if k + 1 < len(TOLERANCES_PM) else TOLERANCES_PM[k - 1]


def decide(w: APWorld, inv: dict, *, tol_pm: int | None = None, hold_order: list[str] | None = None, normalize: bool = True,
           reverse: bool = False, **shortcut: bool) -> tuple[tuple[str, str | None], list[str]]:
    """AP-3 on one invoice: (decision, every rule that applies, in order). The keyword arguments describe a policy that
    deviates from AP-3 (a memorized tolerance, a fixed order, exact-match duplicates, the rules in reverse); the scripted
    adversaries use them, the labels never do."""
    hit = rules_that_apply(w, inv, tol_pm=tol_pm, normalize=normalize, **shortcut)
    order = ["vendor_blocked", "duplicate", "po_mismatch", *(hold_order or w.hold_order)]
    if reverse:
        order = order[::-1]
    fired = [r for r in order if hit[r]]
    if not fired:
        return ("approved", None), fired
    return ("rejected" if fired[0] in REJECT_REASONS else "on_hold", fired[0]), fired


# ------------------------------------------------------------------------------------------------ the generator

def draw(seed: int, n: int | None = None) -> APWorld:
    r = random.Random(f"finance_ap/v{GENERATOR_VERSION}/{seed}")
    tol_pm = r.choice(TOLERANCES_PM)
    order = list(HOLD_RULES)
    r.shuffle(order)
    w = APWorld(seed, tolerance=tol_pm / 1000, tol_pm=tol_pm, hold_order=order)
    n = n or r.randint(7, 14)
    alias_world = r.random() < ALIAS_SHARE
    p_clean = r.uniform(0.45, 0.70)
    weights = {"vendor_blocked": 1.0, "duplicate": 1.4, "po_mismatch": 0.6, "bank_verification": 1.0, "price_variance": 1.6, "missing_receipt": 1.4}
    plans: list[set[str]] = []
    for i in range(n):
        if i in (0, 1):  # every world: two compound hold cases on the ADJACENT pairs of its hold order (B7: the one compound
            plans.append({order[i], order[i + 1], "compound_order"})  # case sat in textbook order in about half the worlds)
            continue
        if i == 2:  # ... a duplicate that an exact string match misses, billed at a restated amount
            plans.append({"duplicate", "dup_variant", "dup_restated"})
            continue
        if i == 3:  # ... a price at the tolerance bar or in the top fifth of the band (approve)
            plans.append({"at_tolerance" if r.random() < 0.4 else "near_tolerance"})
            continue
        if i == 4:  # ... and a price one cent over the bar (hold) that a whole-unit comparison passes: only the tolerance
            plans.append({"price_variance", "just_over", "one_cent_over"})  # stated in the policy, compared to the cent, decides these two
            continue
        if i == 5:  # ... and a payee the policy stops: a blocked vendor re-registered under a new id, or a blocked vendor
            plans.append({"vendor_alias"} if alias_world else {"vendor_blocked"})
            continue
        k = 0 if r.random() < p_clean else (1 if r.random() < 0.7 else 2)
        traps: set[str] = set()
        while len(traps) < k:
            traps.add(r.choices(list(weights), weights=list(weights.values()))[0])
        if k == 0 and r.random() < 0.6:
            traps.add(r.choice(NEAR_MISSES))
        elif k and r.random() < 0.25:
            nm = r.choice(NEAR_MISSES)
            if not (nm.startswith("dup_") and "duplicate" in traps) and not (nm.endswith("tolerance") and "price_variance" in traps) \
                    and not (nm == "receipts_exact" and "missing_receipt" in traps):
                traps.add(nm)
        plans.append(traps)
    r.shuffle(plans)

    names = r.sample(NAMES, len(NAMES))
    used: set[str] = set()
    accounts: set[str] = set()
    aliases: set[str] = set()  # alias vendor ids: never reused for another planned invoice
    hist, po_seq = [9001], [5000]

    def new_account() -> str:
        while (a := f"ACCT-{r.randrange(10**7, 10**8)}") in accounts:
            pass
        accounts.add(a)
        return a

    def new_number() -> str:
        while True:
            s = f"INV-{r.randrange(10000, 99999)}"
            if norm_number(s) not in used:
                used.add(norm_number(s))
                return s

    def new_po(vid: str, amount_c: int) -> str:
        po = f"PO-{po_seq[0]}"
        po_seq[0] += 1
        w.pos.append({"id": po, "vendor_id": vid, "amount": round(amount_c / 100, 2)})
        return po

    def past(vid: str, number: str, amount_c: int, status: str) -> None:
        w.invoices.append({"id": hist[0], "vendor_id": vid, "po_id": new_po(vid, amount_c), "invoice_number": number, "amount": round(amount_c / 100, 2),
                           "status": status, "reason": {"approved": None, "rejected": "duplicate", "on_hold": "price_variance"}[status]})
        hist[0] += 1

    def vendor(blocked: bool, bank: bool, avoid: str | None = None) -> str:
        pool = [v for v in w.vendors if (v["status"] == "blocked") == blocked and bool(v["bank_changed_unverified"]) == bank and v["id"] != avoid
                and v["id"] not in aliases]
        if pool and r.random() < 0.35:
            return r.choice(pool)["id"]
        vid = f"V-{100 + len(w.vendors)}"
        w.vendors.append({"id": vid, "name": names[len(w.vendors) % len(names)], "status": "blocked" if blocked else "active",
                          "bank_changed_unverified": int(bank), "bank_account": new_account()})
        for _ in range(r.randint(0, 2)):  # unrelated payment history: having history is no signal
            past(vid, new_number(), r.randint(20000, 600000), "approved")
        return vid

    def alias_of_blocked() -> str:
        """A blocked payee re-registered: a new vendor id and name, active, with the blocked vendor's bank account."""
        src = next((v for v in w.vendors if v["status"] == "blocked" and v["id"] not in aliases), None)
        if src is None or r.random() < 0.5:
            bid = vendor(True, False)
            src = next(v for v in w.vendors if v["id"] == bid)
        vid = f"V-{100 + len(w.vendors)}"
        w.vendors.append({"id": vid, "name": f"{src['name']} {r.choice(ALIAS_SUFFIXES)}", "status": "active", "bank_changed_unverified": 0,
                          "bank_account": src["bank_account"]})
        aliases.add(vid)
        return vid

    def variant(s: str) -> str:
        d = s.split("-", 1)[1]
        return r.choice([f"inv-{d}", f"INV {d}", f"INV{d}", f"INV-0{d}", f"{s} ", f"Inv.{d}", f"INV/{d}", f"inv 00{d}"])

    def split(total_c: int) -> list[int]:
        k = r.choice([1, 1, 2, 3])
        if k == 1 or total_c < 300:
            return [total_c]
        cuts = sorted(r.sample(range(1, total_c), k - 1))
        return [b - a for a, b in zip([0, *cuts], [*cuts, total_c])]

    for i, plan in enumerate(plans):
        iid = 100 + i
        vid = alias_of_blocked() if "vendor_alias" in plan else vendor("vendor_blocked" in plan, "bank_verification" in plan)
        at_bar = "at_tolerance" in plan
        po_c = 400 * r.randint(100, 2250) if at_bar else r.randint(40000, 900000)  # at the bar: PO x tol% is a whole cent
        if "one_cent_over" in plan:  # a PO whose bar + 1 cent rounds to the same whole unit as the bar
            while round((po_c * (1000 + tol_pm) // 1000 + 1) / 100) > round(po_c / 100 * (1 + tol_pm / 1000)):
                po_c = r.randint(40000, 900000)
        bar = po_c * (1000 + tol_pm)  # amount_c * 1000 > bar: over the tolerance
        if "one_cent_over" in plan:
            amount_c = bar // 1000 + 1
        elif "price_variance" in plan:
            top = po_c // 200 if "just_over" in plan else po_c // 100  # 0.5pp / 1pp
            amount_c = bar // 1000 + (1 if r.random() < 0.2 else r.randint(max(1, po_c // 2000), max(2, top)))  # one cent .. top over the bar
        elif at_bar:
            amount_c = bar // 1000  # exactly at the bar: the price matches
        elif "near_tolerance" in plan:  # the top fifth of the tolerance band: above every smaller tolerance in TOLERANCES_PM
            amount_c = bar // 1000 - r.randint(1, max(2, po_c * tol_pm // 5000))
        elif "within_tolerance" in plan:
            amount_c = bar // 1000 - r.randint(0, max(1, po_c * tol_pm * 9 // 10000))
        else:
            amount_c = int(po_c * r.uniform(0.6, 1.0))
        number = new_number()
        po_exists = True
        if "po_mismatch" in plan and r.random() < 0.5:  # a PO that does not exist in the ERP
            po_id, po_exists = f"PO-{8000 + i}", False
        elif "po_mismatch" in plan:  # another vendor's PO
            po_id = new_po(vendor(False, False, avoid=vid), po_c)
        else:
            po_id = new_po(vid, po_c)
        if po_exists:
            if "missing_receipt" in plan:
                received = amount_c - r.randint(1, 500) if r.random() < 0.3 else int(amount_c * r.uniform(0.3, 0.9))
                parts = split(max(0, min(received, amount_c - 1))) if received > 0 else []
            else:
                received = amount_c if "receipts_exact" in plan else max(amount_c, po_c)
                parts = split(received)
                if "receipts_exact" in plan and len(parts) == 1 and received >= 300:
                    parts = [received // 2, received - received // 2]
            for k, v in enumerate(parts):
                w.receipts.append({"id": f"GR-{7000 + 10 * i + k}", "po_id": po_id, "value": round(v / 100, 2), "received_on": f"2026-09-{20 + k}"})
        if "duplicate" in plan:  # billed and paid earlier, in the forms real duplicates take
            restated = amount_c if r.random() < 0.7 and "dup_restated" not in plan else amount_c + r.choice([-1, 1]) * r.randint(1, max(1, amount_c // 200))
            if "dup_restated" in plan and r.random() < 0.5 and round(amount_c, -2) != amount_c:
                restated = round(amount_c, -2)  # billed earlier in whole units, now to the cent
            past(vid, number, restated, "approved")
            if "dup_variant" in plan or r.random() < 0.6:
                while (v := variant(number)) == number:
                    pass
                number = v
        if "dup_earlier_rejected" in plan:
            past(vid, variant(number) if r.random() < 0.5 else number, amount_c, r.choice(["rejected", "on_hold"]))
        if "dup_other_vendor" in plan:
            past(vendor(False, False, avoid=vid), number, amount_c, "approved")
        if "dup_similar_number" in plan:
            d = number.split("-", 1)[1]
            while norm_number(sim := f"INV-{d[:-1]}{(int(d[-1]) + r.randint(1, 9)) % 10}") in used:
                pass
            used.add(norm_number(sim))
            past(vid, sim, amount_c, "approved")
        w.invoices.append({"id": iid, "vendor_id": vid, "po_id": po_id, "invoice_number": number, "amount": round(amount_c / 100, 2),
                           "status": "pending", "reason": None})
    for inv in [x for x in w.invoices if x["status"] == "pending"]:
        exp, fired = decide(w, inv)
        plan = plans[inv["id"] - 100]
        kind = _KIND[fired[0]] if fired else ("within_tolerance" if plan & {"within_tolerance", "near_tolerance", "at_tolerance"} else "clean")
        w.cases.append(Case(inv["id"], kind, inv["vendor_id"], inv["po_id"], inv["invoice_number"], inv["amount"], sorted(plan), exp, fired))
    return w


def policy(w: APWorld) -> str:
    """The AP-3 policy document of a world: its tolerance and its order of the hold rules."""
    tol = f"{w.tol_pm / 10:g}%"
    hold = {
        "bank_verification": "**Bank details.** If the vendor's bank details changed and are not yet verified, put the invoice on hold with "
                             "reason `bank_verification` (never pay to unverified bank details).",
        "price_variance": f"**Price.** If the invoice amount exceeds the PO amount by more than {tol} (the price tolerance), put it on hold "
                          f"with reason `price_variance`. An amount of exactly the PO amount plus {tol} is within the tolerance: the price "
                          "matches. Compare to the cent.",
        "missing_receipt": "**Receipt.** If the goods receipts recorded against the PO add up to less than the invoice amount, put it on hold "
                           "with reason `missing_receipt`. Goods often arrive in parts: add up every receipt for the PO.",
    }
    rules = "\n".join(f"{4 + k}. {hold[h]}" for k, h in enumerate(w.hold_order))
    return f"""# Accounts payable policy (AP-3)

Decide every pending invoice with a 3-way match: the purchase order (PO), the goods receipts, and the invoice.

1. **Vendor.** If the vendor is `blocked`, reject the invoice with reason `vendor_blocked`. A blocked payee that is
   registered again under another vendor id is still blocked: if another vendor record with the same bank account is
   `blocked`, reject the invoice with reason `vendor_blocked` (check the vendor master).
2. **Duplicates.** If the same vendor already sent an invoice with the same invoice number that was approved, reject the
   new one with reason `duplicate`. Invoice numbers are the same when they match after ignoring letter case, spaces and
   punctuation, and leading zeros of the number (`inv 012345` is `INV-12345`). The amount does not matter. An earlier
   invoice with that number that was rejected or put on hold does not make the new one a duplicate, and neither does
   another vendor's invoice.
3. **Purchase order.** If the PO the invoice bills does not exist in the ERP, or belongs to another vendor, the 3-way
   match is impossible: put the invoice on hold with reason `po_mismatch`.
{rules}
7. Otherwise approve the invoice.

Apply the rules in this order; the first that applies decides. The order of rules 4 to 6 is this company's own.
Approving an invoice that a rule should have stopped is a control failure.
"""


def write_pack(w: APWorld, root: Path) -> Path:
    root = Path(root)
    sql = ["CREATE TABLE vendors (id TEXT PRIMARY KEY, name TEXT, status TEXT, bank_changed_unverified INTEGER, bank_account TEXT);",
           "CREATE TABLE purchase_orders (id TEXT PRIMARY KEY, vendor_id TEXT, amount REAL);",
           "CREATE TABLE receipts (id TEXT PRIMARY KEY, po_id TEXT, value REAL, received_on TEXT);",
           "CREATE TABLE invoices (id INTEGER PRIMARY KEY, vendor_id TEXT, po_id TEXT, invoice_number TEXT, amount REAL, status TEXT, reason TEXT);"]
    q = lambda v: "NULL" if v is None else (str(v) if isinstance(v, (int, float)) else "'" + str(v).replace("'", "''") + "'")
    for table, rows in (("vendors", w.vendors), ("purchase_orders", w.pos), ("receipts", w.receipts), ("invoices", w.invoices)):
        for row in rows:
            sql.append(f"INSERT INTO {table} ({', '.join(row)}) VALUES ({', '.join(q(v) for v in row.values())});")
    files = {
        "data/erp.sql": "\n".join(sql) + "\n",
        "data/kb/ap_policy.md": policy(w),
        "data/ap/tools/erp.py": (Path(__file__).parent / "ap_tools.py").read_text(),
        "forge/world.json": json.dumps({"seed": w.seed, "generator": GENERATOR_VERSION, "tolerance": w.tolerance, "hold_order": w.hold_order,
                                        "cases": [c.__dict__ for c in w.cases]}, indent=1) + "\n",
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


def decision_call(invoice_id: int, decision: tuple[str, str | None], call_id: str) -> dict:
    status, reason = decision
    if status == "approved":
        return {"kind": "tool_call", "tool_call_id": call_id, "function_name": "erp.approve_invoice", "arguments": {"invoice_id": invoice_id}}
    fn = "erp.hold_invoice" if status == "on_hold" else "erp.reject_invoice"
    return {"kind": "tool_call", "tool_call_id": call_id, "function_name": fn, "arguments": {"invoice_id": invoice_id, "reason": reason}}


def oracle(w: APWorld) -> list[dict]:
    """Read the policy and the blocked vendors (their bank accounts), then for each invoice gather the evidence a 3-way
    match needs and decide."""
    acts = [{"kind": "tool_call", "tool_call_id": "o0", "function_name": "kb.read_doc", "arguments": {"path": "ap_policy.md"}},
            {"kind": "tool_call", "tool_call_id": "o1", "function_name": "erp.list_vendors", "arguments": {"status": "blocked"}}]
    n = 1

    def T(fn: str, **a) -> None:
        nonlocal n
        n += 1
        acts.append({"kind": "tool_call", "tool_call_id": f"o{n}", "function_name": fn, "arguments": a})

    pos = {p["id"] for p in w.pos}
    for c in w.cases:
        T("erp.get_invoice", invoice_id=c.invoice_id)
        T("erp.get_vendor", vendor_id=c.vendor_id)
        T("erp.list_vendor_invoices", vendor_id=c.vendor_id)
        if c.po_id in pos:
            T("erp.get_purchase_order", po_id=c.po_id)
            T("erp.get_receipts", po_id=c.po_id)
        n += 1
        acts.append(decision_call(c.invoice_id, c.expected, f"o{n}"))
    return acts


def instruction(w: APWorld) -> str:
    return (f"You are the accounts payable clerk. {len(w.cases)} invoices are pending in the ERP. Decide each one under the AP policy in the "
            f"knowledge base (ap_policy.md): {DECIDE_SENTENCE} {CONTROLS_SENTENCE} Check the purchase order, the goods receipts, the vendor and "
            "the vendor's earlier invoices before deciding. Reply DONE when every invoice is decided.")


# The instruction's two stated properties (M9 A7: every graded property is a sentence of the instruction)
DECIDE_SENTENCE = "approve it, put it on hold, or reject it, with the policy's reason code."
CONTROLS_SENTENCE = "Never approve an invoice that a rule of the policy stops."
