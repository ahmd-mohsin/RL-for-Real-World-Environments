"""``toy/helpdesk_backlog``: a support shift of hundreds of tickets (DESIGN §10.4). One episode is thousands of tool calls,
and its full trajectory (every observation and every model reply, before any compaction) exceeds 1M tokens at the
``xl`` preset.

What makes the horizon *long*, not just big:

- **Revisits after hundreds of steps.** ``awaiting`` tickets must not be acted on until the customer's reply arrives as
  a message, hours of simulated time later. The ticket is then handled by what the reply says (refund, or close).
  Acting early is wrong, forgetting it is wrong.
- **A rule that changes mid-episode.** At ``amend_at`` the refund policy is amended (the approval threshold drops from
  50 to 25 USD): the knowledge-base document is rewritten and an announcement arrives. A refund is judged by the rule
  in force *when it was issued*, so a policy whose context dropped the announcement fails every later small refund.
- **Long-range consistency.** ``repeat`` tickets ask again about an order that an earlier ticket refunds (or that was
  refunded before the shift). A second refund is an over-refund.
- **Bulk context.** Every ticket carries a customer thread of several messages, and the knowledge base holds filler
  documents next to the two that matter.

Ground truth is per ticket (``expected``), so the grade decomposes exactly into per-ticket subgoals, and the dense
progress reward telescopes to the final reward (``task.py``, ``crucible.verify.progress``).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from crucible_pack_toy import PACK_ROOT

T0 = "2026-05-04T08:00:00Z"
STEP_DT_S = 60
PRESETS = {"tiny": 12, "small": 40, "medium": 120, "xl": 1200, "xxl": 2400}  # xl: > 1M trajectory tokens even with no reasoning tokens
KINDS = ("dup_small", "damaged", "in_transit", "info", "repeat", "awaiting")
WEIGHTS = (0.22, 0.24, 0.12, 0.14, 0.12, 0.16)

FIRST = ["Mara", "Jon", "Ines", "Tariq", "Lena", "Omar", "Priya", "Felix", "Hana", "Diego", "Zoe", "Ravi", "Ada", "Kofi", "Mei", "Lars", "Sana", "Ivo"]
LAST = ["Okafor", "Lindqvist", "Duarte", "Haddad", "Novak", "Sato", "Mensah", "Keller", "Rossi", "Iyer", "Brandt", "Costa", "Nair", "Ueda"]
ITEMS = ["blender", "desk lamp", "headphones", "coffee grinder", "backpack", "monitor stand", "kettle", "air fryer", "keyboard", "rain jacket",
         "office chair", "router", "stand mixer", "e-reader", "camping stove", "label printer"]
FILLER_TOPICS = ["shipping", "warranty", "gift_cards", "loyalty", "privacy", "accessibility", "bulk_orders", "returns_labels", "store_hours",
                 "payment_methods", "price_matching", "international", "subscriptions", "repairs"]

POLICY_V1 = """# Refund policy FIN-7 (current)

1. A refund is at most the order amount, and an order is refunded at most once in total.
2. Duplicate charges: refund the duplicate (the full order amount).
3. **Refunds above {t1} USD need the supervisor's approval.** Ask the supervisor, wait for the approval, and record the approver's name in `approved_by`.
4. Orders still in transit are not refundable: add a note and set the ticket to `escalated` for a carrier trace.
5. Damaged items: refund in full.
6. A ticket waiting for the customer's confirmation stays `pending`; act only on the customer's reply.
7. Every ticket you handle gets a note.
8. A second ticket about the same order: any refund belongs on the original ticket only; resolve the second ticket with a note.
"""
POLICY_V2 = POLICY_V1.replace("FIN-7 (current)", "FIN-7a (current, amended {when})").replace("above {t1} USD", "above {t2} USD")


def _text_policy(t1: float, t2: float | None = None, when: str = "") -> str:
    return (POLICY_V2 if t2 is not None else POLICY_V1).format(t1=f"{t1:g}", t2=f"{t2:g}" if t2 is not None else "", when=when)


@dataclass
class Ticket:
    id: int
    kind: str
    customer: str
    order: str
    amount: float
    subject: str
    thread: list[dict]
    expected: dict  # {"refund": amount|None, "status": ..., "note": True, "wait_for": event_id|None}
    follow_up: dict | None = None


@dataclass
class Backlog:
    seed: int
    n: int
    supervisor: str
    t1: float
    t2: float
    amend_at_h: float
    customers: dict[str, dict] = field(default_factory=dict)
    orders: dict[str, dict] = field(default_factory=dict)
    tickets: list[Ticket] = field(default_factory=list)
    prior_refunds: list[dict] = field(default_factory=list)

    @property
    def amend_at(self) -> str:
        return (datetime.fromisoformat(T0) + timedelta(hours=self.amend_at_h)).strftime("%Y-%m-%dT%H:%M:%S")  # UTC, as the tools record it

    def threshold_at(self, iso: str) -> float:
        return self.t2 if iso >= self.amend_at else self.t1


def _sentence(rng: random.Random, item: str, order: str, kind: str) -> list[str]:
    opener = rng.choice(["Hi,", "Hello support,", "Good morning,", "Hey there,", "Dear team,"])
    body = {
        "dup_small": [f"I was charged twice for order {order} ({item}).", "My bank statement shows two identical charges on the same day.",
                      "Please reverse the duplicate charge."],
        "damaged": [f"The {item} from order {order} arrived damaged.", "The box was crushed and the item does not work.",
                    "I would like a full refund, I do not want a replacement."],
        "in_transit": [f"I ordered a {item} (order {order}) and it has not arrived.", "The tracking page still says it is on its way.",
                       "Can I just get my money back now?"],
        "info": [f"Could you send me an invoice copy for order {order}?", "I need it for my company's expense report.", "No refund needed, just the document."],
        "repeat": [f"Following up on order {order} ({item}).", "I asked about this before and want to be sure it is handled.", "Please refund it if you have not."],
        "awaiting": [f"Order {order} ({item}) shows as delivered but I cannot find it.", "I am checking with my neighbours and the building office.",
                     "I will write again once I know; please do nothing until then."],
    }[kind]
    chatter = [rng.choice(["I have been a customer for years.", "This is the first time something like this happened.", "Thanks for your help.",
                           "I attached nothing because the form would not let me.", "Please reply by email if possible.",
                           "My partner placed a similar order last month without problems.", "I am travelling next week."]) for _ in range(rng.randint(1, 3))]
    return [opener, *body, *chatter]


def draw(seed: int = 0, n: int = 120, *, amend_at_h: float | None = None) -> Backlog:
    rng = random.Random(seed)
    sup = rng.choice(["Sam Lee", "Rae Kim", "Noor Haddad", "Theo Brandt"])
    est_steps = int(n * 3.6)  # the oracle's length; the amendment and the follow-ups fall inside it
    b = Backlog(seed=seed, n=n, supervisor=sup, t1=50.0, t2=25.0, amend_at_h=amend_at_h if amend_at_h is not None else round(est_steps * STEP_DT_S / 3600 * 0.45, 2))
    used: set[str] = set()
    n_cust = max(4, n // 3)
    for i in range(n_cust):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if name in used:  # more customers than first x last names: disambiguate with a middle initial
            name = f"{name.split()[0]} {chr(65 + i % 26)}. {name.split()[1]}-{i}"
        used.add(name)
        cid = f"C-{1000 + i}"
        b.customers[cid] = {"id": cid, "name": name, "tier": rng.choice(["standard", "standard", "gold"])}
    cids = list(b.customers)
    refundable: list[tuple[str, int]] = []  # (order, ticket) refunded by an earlier dup/damaged ticket (for repeat tickets)
    oid = 50000
    awaiting_idx = 0
    for k in range(n):
        tid = 100 + k
        kind = rng.choices(KINDS, WEIGHTS)[0]
        if kind == "repeat" and not refundable and not b.prior_refunds:
            kind = "dup_small"
        cust = rng.choice(cids)
        item = rng.choice(ITEMS)
        if kind == "repeat":
            prev = rng.choice(refundable) if refundable and rng.random() < 0.7 else None
            origin = None
            if prev is None:  # an order refunded before the shift
                oid += 1
                order = f"O-{oid}"
                amt = round(rng.uniform(20, 300), 2)
                b.orders[order] = {"id": order, "customer": cust, "amount": amt, "status": "delivered"}
                b.prior_refunds.append({"ticket_id": 9000 + k, "order_id": order, "amount_usd": amt, "approved_by": sup if amt > b.t1 else None,
                                        "created_at": "2026-04-20T10:00:00"})
            else:
                order, origin = prev
                cust = b.orders[order]["customer"]
                amt = b.orders[order]["amount"]
        else:
            oid += 1
            order = f"O-{oid}"
            amt = {"dup_small": rng.uniform(12, 49), "damaged": rng.uniform(26, 380), "in_transit": rng.uniform(40, 900), "info": rng.uniform(15, 400),
                   "awaiting": rng.uniform(20, 260)}[kind]
            amt = round(amt, 2)
            b.orders[order] = {"id": order, "customer": cust, "amount": amt, "status": "in_transit" if kind == "in_transit" else "delivered"}
        msgs = _sentence(rng, item, order, kind)
        if kind == "repeat":
            msgs.insert(2, f"This is about my earlier ticket #{origin}." if origin else "I believe you refunded it in April, please check.")
        thread = [{"from": b.customers[cust]["name"], "at": f"2026-05-0{rng.randint(1, 3)}T{rng.randint(8, 20):02d}:{rng.randint(0, 59):02d}:00Z",
                   "text": " ".join(msgs)}]
        for _ in range(rng.randint(1, 4)):  # earlier back-and-forth: bulk context with no new facts
            thread.append({"from": rng.choice(["support bot", b.customers[cust]["name"]]), "at": "2026-05-03T12:00:00Z",
                           "text": rng.choice(["We received your message and will get back to you.", "Any update on this?", "Thank you for your patience.",
                                               "Just checking whether you need anything else from me.", "Your case is in the queue."])})
        subj = {"dup_small": f"Charged twice for {order}", "damaged": f"{item.title()} arrived damaged ({order})", "in_transit": f"Refund for {order}?",
                "info": f"Invoice for {order}", "repeat": f"Again about {order}", "awaiting": f"{order} delivered but missing"}[kind]
        exp: dict[str, Any] = {"note": True}
        fu = None
        if kind in ("dup_small", "damaged"):
            exp.update(refund=amt, status="resolved")
            refundable.append((order, tid))
        elif kind in ("in_transit",):
            exp.update(refund=None, status="escalated")
        elif kind in ("info", "repeat"):
            exp.update(refund=None, status="resolved")
        else:
            awaiting_idx += 1
            found = rng.random() < 0.4
            fid = f"followup_{tid}"
            at_h = round(est_steps * STEP_DT_S / 3600 * (0.15 + 0.7 * rng.random()), 2)
            fu = {"id": fid, "after_h": at_h, "text": f"[customer {cust}] Ticket {tid}: " + ("I found the package at the building office, please close my request."
                                                                                               if found else "It is definitely lost. Please refund order " + order + " now."),
                  "found": found}
            exp.update(refund=None if found else amt, status="resolved", wait_for=fid)
        b.tickets.append(Ticket(tid, kind, cust, order, amt, subj, thread, exp, fu))
    return b


def _q(s: Any) -> str:
    return "NULL" if s is None else ("'" + str(s).replace("'", "''") + "'" if isinstance(s, str) else f"{s}")


def _filler(rng: random.Random, topic: str) -> str:
    paras = []
    for i in range(rng.randint(10, 18)):
        paras.append(f"## {topic.replace('_', ' ').title()} {i + 1}\n\n" + " ".join(rng.choice([
            "Customers can find this information in their account settings.", "Our team reviews each case within two business days.",
            "This section does not change any refund rule.", "Exceptions are listed in the current refund policy only.",
            "Store credit is offered at the customer's request.", "Contact the logistics desk for carrier questions.",
            "Keep a record of every interaction in the ticket notes.", "Promotional terms are published separately each quarter."]) for _ in range(8)))
    return f"# {topic.replace('_', ' ').title()}\n\n" + "\n\n".join(paras) + "\n"


def write_pack(b: Backlog, root: Path) -> Path:
    """The forged pack: world spec, schema + rows, knowledge base, tools, grounding records, scenario."""
    root = Path(root)
    rng = random.Random(b.seed + 1)
    schema = (PACK_ROOT / "data" / "tickets.sql").read_text().split("INSERT INTO")[0].strip()
    schema = schema.replace("  opened_at TEXT NOT NULL,", "  opened_at TEXT NOT NULL,\n  thread TEXT,")
    lines = [schema]
    lines.append("INSERT INTO customers VALUES " + ", ".join(f"({_q(c['id'])}, {_q(c['name'])}, {_q(c['tier'])})" for c in b.customers.values()) + ";")
    lines.append("INSERT INTO orders VALUES " + ",\n  ".join(f"({_q(o['id'])}, {_q(o['customer'])}, {o['amount']:.2f}, {_q(o['status'])})" for o in b.orders.values()) + ";")
    lines.append("INSERT INTO tickets(id, customer_id, subject, status, opened_at, thread, resolved_at) VALUES " + ",\n  ".join(
        f"({t.id}, {_q(t.customer)}, {_q(t.subject)}, 'open', '2026-05-04T07:00:00Z', {_q(json.dumps(t.thread))}, NULL)" for t in b.tickets) + ";")
    if b.prior_refunds:
        lines.append("INSERT INTO refunds(ticket_id, order_id, amount_usd, approved_by, created_at) VALUES " + ", ".join(
            f"({r['ticket_id']}, {_q(r['order_id'])}, {r['amount_usd']:.2f}, {_q(r['approved_by'])}, {_q(r['created_at'])})" for r in b.prior_refunds) + ";")
    files: dict[str, str] = {
        "data/tickets.sql": "\n".join(lines) + "\n",
        "data/kb/refund_policy.md": _text_policy(b.t1),
        "data/kb/escalation.md": (PACK_ROOT / "data" / "kb" / "escalation.md").read_text(),
        "data/helpdesk/tools/backlog.py": (Path(__file__).parent / "backlog_tools.py").read_text(),
        "data/customers.jsonl": "".join(json.dumps({"id": f"acct#{c['id']}", "source": "account", "author": c["id"], "role": "customer", "time": "2026-05-01T00:00:00Z",
                                                    "text": f"I am {c['name']} ({c['id']})."}) + "\n" for c in b.customers.values()),
        "forge/backlog.json": json.dumps({"seed": b.seed, "n": b.n, "supervisor": b.supervisor, "t1": b.t1, "t2": b.t2, "amend_at": b.amend_at,
                                          "tickets": [t.__dict__ for t in b.tickets]}, indent=1, default=str) + "\n",
    }
    for topic in FILLER_TOPICS:
        files[f"data/kb/{topic}.md"] = _filler(rng, topic)
    events = [{"id": "policy_amendment", "when": {"after": f"{int(b.amend_at_h * 3600)}s"}, "actor": "ops",
               "action": {"type": "tool", "tool": "kb.write_doc", "args": {"path": "refund_policy.md", "content": _text_policy(b.t1, b.t2, b.amend_at)}}},
              {"id": "policy_amendment_notice", "when": {"after": f"{int(b.amend_at_h * 3600)}s"}, "actor": "ops",
               "action": {"type": "nl", "to": "agent", "text": f"Policy update, effective now: refunds above {b.t2:g} USD (was {b.t1:g}) need the supervisor's "
                                                                "approval. The knowledge base has the amended FIN-7a."}}]
    for t in b.tickets:
        if t.follow_up:
            events.append({"id": t.follow_up["id"], "when": {"after": f"{int(t.follow_up['after_h'] * 3600)}s"}, "actor": "ops",
                           "action": {"type": "nl", "to": "agent", "text": t.follow_up["text"]}})
    world = {
        "schema_version": "crucible.world/v1", "world": "toy/helpdesk_backlog",
        "description": f"A support shift: {b.n} tickets, a mid-shift policy amendment, customer replies arriving hours later (seed {b.seed}).",
        "backend": "local", "backend_params": {"bundles": [{"src": "data/helpdesk", "dest": "helpdesk"}]},
        "apps": [
            {"id": "tickets", "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": "helpdesk/db/tickets.db"}},
             "params": {"schema_sql": "data/tickets.sql", "tools_module": "helpdesk/tools/backlog.py", "transport": "inprocess"}, "description": "ticketing system of record"},
            {"id": "kb", "template": "common/doc_library", "state": {"adapter": "fs", "params": {"root": "kb"}}, "params": {"seed_dir": "data/kb", "writable": True},
             "description": "policy and help-center documents"},
        ],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["tickets.*", "kb.list_docs", "kb.read_doc", "kb.search_docs"]},
                   {"id": "supervisor", "kind": "system"}, {"id": "ops", "kind": "system", "tools": ["kb.write_doc"]}],
        "clock": {"mode": "step", "t0": T0, "step_dt_s": STEP_DT_S},
        "events": events,
        "policies": [{"id": "refund_policy", "app": "kb", "path": "refund_policy.md", "applies_to": ["agent", "supervisor"]}],
    }
    files["worlds/helpdesk_backlog.yaml"] = yaml.safe_dump(world, sort_keys=False, width=200)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def instruction(b: Backlog) -> str:
    return (f"You are on the support shift of an online shop. The backlog has {b.n} open tickets (list them page by page). Work through every one "
            "of them until each is in its correct final state under the current refund policy in the knowledge base (refund_policy.md). Read each "
            "ticket and, before refunding, check the order's refund history. Ask the supervisor for approval whenever the policy requires it and "
            "record the approver's name. Some customers asked you to wait for their reply: keep those tickets pending and act only when their "
            "message arrives. The policy can be amended during the shift; announcements arrive as messages and the knowledge base is updated. "
            "When every ticket is handled, reply DONE.")


def oracle(b: Backlog) -> list[dict]:
    """A correct shift, step-timed: each action is one World step of ``STEP_DT_S``; awaiting tickets are parked (pending + note) and
    handled after their reply has fired; every refund above the *lower* threshold asks for approval (always allowed), so the
    oracle is right on both sides of the amendment."""
    acts: list[dict] = []
    n = 0

    def T(fn: str, **a: Any) -> None:
        nonlocal n
        n += 1
        acts.append({"kind": "tool_call", "tool_call_id": f"o{n}", "function_name": fn, "arguments": a})

    def M(text: str) -> None:
        acts.append({"kind": "message", "to": "supervisor", "text": text})

    def refund(t: Ticket, amt: float) -> None:
        if amt > min(b.t1, b.t2):
            M(f"Please approve a refund of {amt:.2f} USD for ticket {t.id}, order {t.order}.")
            T("tickets.issue_refund", ticket_id=t.id, order_id=t.order, amount_usd=amt, approved_by=b.supervisor)
        else:
            T("tickets.issue_refund", ticket_id=t.id, order_id=t.order, amount_usd=amt)

    def fired(fu: dict) -> bool:
        return len(acts) * STEP_DT_S > fu["after_h"] * 3600 + 2 * STEP_DT_S  # strictly after the reply's step (events fire between steps)

    parked: list[Ticket] = []
    for t in b.tickets:
        for p in [p for p in parked if fired(p.follow_up)]:
            parked.remove(p)
            _finish_awaiting(p, refund, T)
        if t.kind == "awaiting":
            T("tickets.add_note", ticket_id=t.id, body="Waiting for the customer's confirmation before acting.")
            T("tickets.set_status", ticket_id=t.id, status="pending")
            parked.append(t)
            continue
        if t.expected["refund"] is not None:
            refund(t, t.expected["refund"])
        T("tickets.add_note", ticket_id=t.id, body={"in_transit": "Order still in transit: escalated for a carrier trace.", "info": "Invoice copy sent.",
                                                    "repeat": "This order has already been refunded; no second refund."}.get(t.kind, "Refunded per policy FIN-7."))
        T("tickets.set_status", ticket_id=t.id, status=t.expected["status"])
    while parked:
        ready = [p for p in parked if fired(p.follow_up)]
        if not ready:
            T("kb.list_docs")  # waiting for a reply: any read-only step advances the clock
            continue
        for p in ready:
            parked.remove(p)
            _finish_awaiting(p, refund, T)
    return acts


def _finish_awaiting(t: Ticket, refund, T) -> None:
    if t.expected["refund"] is not None:
        refund(t, t.expected["refund"])
    T("tickets.add_note", ticket_id=t.id, body="Customer replied; handled as requested.")
    T("tickets.set_status", ticket_id=t.id, status="resolved")


def diligent(b: Backlog, *, reread_policy_every: int = 40) -> list[dict]:
    """The oracle's decisions with the reads a careful agent makes first: the policy, each listing page, every ticket, the
    order's refund history before any refund, and the policy again from time to time. Used to size real trajectories and
    as an API-free long-horizon policy."""
    out: list[dict] = []
    n = 0

    def R(fn: str, **a: Any) -> None:
        nonlocal n
        n += 1
        out.append({"kind": "tool_call", "tool_call_id": f"r{n}", "function_name": fn, "arguments": a})

    R("kb.read_doc", path="refund_policy.md")
    for off in range(0, b.n, 25):
        R("tickets.list_tickets", status="open", offset=off, limit=25)
    by_id = {t.id: t for t in b.tickets}
    handled = 0
    for a in oracle(b):
        tid = (a.get("arguments") or {}).get("ticket_id")
        first = a.get("function_name") in ("tickets.issue_refund", "tickets.add_note") or (a["kind"] == "message")
        if tid is None and a["kind"] == "message":
            tid = int(re.search(r"ticket (\d+)", a["text"]).group(1))
        if first and tid in by_id and by_id[tid] is not None:
            R("tickets.get_ticket", ticket_id=tid)
            R("tickets.get_order", order_id=by_id[tid].order)
            handled += 1
            if handled % reread_policy_every == 0:
                R("kb.read_doc", path="refund_policy.md")
            by_id[tid] = None if by_id[tid].kind != "awaiting" else by_id[tid]  # awaiting tickets are read again after the reply
            if by_id[tid] is not None and a.get("function_name") == "tickets.add_note" and "Waiting" in a["arguments"].get("body", ""):
                pass
        out.append({**a, "tool_call_id": f"d{len(out)}"} if a["kind"] == "tool_call" else a)
    return out
