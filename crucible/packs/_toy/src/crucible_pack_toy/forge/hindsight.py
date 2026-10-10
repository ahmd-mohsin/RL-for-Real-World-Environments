"""G-hindsight for the toy pack (DESIGN §7.4): recorded ticket cases become tasks graded against what really happened
and **held up**, with the expert's corrected mistakes as trap negatives.

Source: the support-ticket logs (``data/ticket_logs/tickets.jsonl``, ingested by M3, signals by the toy M4 mapper), the
toy stand-in for an enterprise's own workflow records. For each refund case:

- the **inputs** are as at the start: the customer's request, their account and orders, the policies;
- the **answer** is the net outcome after every correction (refunds minus reversals, the approval the supervisor gave,
  the final status). An agent's first action that a supervisor later reversed, flagged or corrected is *not* the answer;
- the expert's **original wrong action** is replayed as a trap-reversion negative, citing the signal that recorded the
  correction (``violation``, ``revert``, ``correction``, ``rework``), so the trap is grounded in a real mistake (A8);
- policy rules a correction revealed are written into the world's knowledge base with the signal as provenance (T1003:
  "annual plans bought within 7 days: refund the price less a 10.00 fee, with approval" is **inferred from one
  supervisor correction** (240.00 -> 230.00); the card records that it rests on a single record).

``python -m crucible_pack_toy.forge.hindsight`` forges the six tasks, accepts them with the M9 subset, and writes
``docs/modules/reports/m8_toy_hindsight.json``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from crucible.accept.contract import accept, summarize
from crucible.connectors.trace_store import TraceStore
from crucible.core.store import Store
from crucible.forge.schema import NegativeSpec, StatedProperty, TaskSpec
from crucible.signals.extract import extract
from crucible_pack_toy import PACK_ROOT
from crucible_pack_toy.forge.env import CURRENT_POLICY, T0, write_pack
from crucible_pack_toy.forge.roles import T, compile_checks, negatives_for, oracle_actions
from crucible_pack_toy.forge.scenario import INTRO, POLICY, SCOPE
from crucible_pack_toy.forge.task import ToyForgedTask

CRUCIBLE = PACK_ROOT.parent.parent
M3 = CRUCIBLE / "data" / "m3" / "store"
LOGS = PACK_ROOT / "data" / "ticket_logs" / "tickets.jsonl"
SUPERVISOR = "Sam Lee"  # the supervisor in the recorded logs ("sam")
EXTRA_POLICY = """# Billing corrections (mined from supervisor corrections in the ticket history)
- Annual plans bought within the last 7 days: refund the price less a 10.00 USD processing fee, with supervisor approval.
- Add-ons used during the billing month are not refundable.
- Charges taken after an account was closed are refunded in full.
"""

# case -> how its held-up outcome maps onto the helpdesk world (customer, order, amount, what held up)
CASES = {
    "T1001": {"order": ("O-5521", 39.50, "delivered"), "held": {"refund": 39.50}, "kind": "duplicate charge"},
    "T1003": {"order": ("O-6240", 240.00, "delivered"), "held": {"refund": 230.00}, "kind": "annual plan bought by mistake",
              "mistake": [T(1, "tickets.issue_refund", ticket_id=None, order_id="O-6240", amount_usd=240.00)]},
    "T1013": {"order": ("O-6199", 19.99, "delivered"), "held": {"refund": 19.99}, "kind": "duplicate subscription",
              "mistake": [T(1, "tickets.set_status", ticket_id=None, status="resolved")]},
    "T1015": {"order": ("O-6089", 89.00, "delivered"), "held": {"refund": 89.00}, "kind": "cancelled before renewal",
              "mistake": [T(1, "tickets.issue_refund", ticket_id=None, order_id="O-6089", amount_usd=89.00)]},
    "T1018": {"order": ("O-6012", 12.00, "delivered"), "held": {"refund": 12.00}, "kind": "billed after closing"},
    "T1021": {"order": ("O-6015", 15.00, "used"), "held": {"refund": None}, "kind": "add-on refund request",
              "mistake": [T(1, "tickets.issue_refund", ticket_id=None, order_id="O-6015", amount_usd=15.00)]},
}


def load_cases() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for line in LOGS.read_text().splitlines():
        r = json.loads(line)
        out.setdefault(r["case"], []).append(r)
    return out


def case_signals() -> tuple[dict[str, list[dict]], set[str]]:
    ts = TraceStore(Store(M3))
    out: dict[str, list[dict]] = {}
    known: set[str] = set()
    for tid in sorted(i for i in ts.ids() if i.startswith("toy/support_ticket:")):
        case = tid.split(":", 1)[1]
        for s in extract(ts.get(tid)).signals:
            known.add(s.id)
            if s.polarity < 0:
                out.setdefault(case, []).append({"id": s.id, "type": s.type, "text": s.text, "rule": s.meta.get("rule")})
    return out, known


def build(seed_ticket: int = 61) -> dict[str, Any]:
    logs = load_cases()
    sigs, known = case_signals()
    customers, orders, tickets, tasks, oracles, negatives = {}, {}, {}, [], {}, {}
    letters = "ABCDEFGH"
    for k, (case, spec) in enumerate(CASES.items()):
        L = letters[k]
        opener = next(e for e in logs[case] if e["type"] == "ticket_opened")
        customers[L] = {"id": f"C-{500 + k}", "name": f"Customer {case}", "tier": "standard"}
        oid, amount, status = spec["order"]
        orders[f"{L}1"] = {"id": oid, "customer": L, "amount": amount, "status": status}
        tid = seed_ticket + k
        tickets[case] = tid
        held = spec["held"]["refund"]
        props = []
        if held is not None:
            props.append(StatedProperty(id="refund", text=f"Resolve the request on ticket {tid} as the policies say, refunding order {oid} where they allow it.",
                                        kind="refund", args={"ticket": tid, "order": oid, "amount": held}))
        else:
            props.append(StatedProperty(id="no_refund", text=f"Resolve the request on ticket {tid} as the policies say, refunding order {oid} only where they allow it.",
                                        kind="no_refund", args={"ticket": tid, "order": oid}))
        props += [StatedProperty(id="note", text=f"Add a note to ticket {tid}.", kind="note", args={"ticket": tid}),
                  StatedProperty(id="status", text=f"Set ticket {tid} to resolved.", kind="status", args={"ticket": tid, "value": "resolved"}),
                  StatedProperty(id="policy", text=POLICY, kind="policy"), StatedProperty(id="scope", text=SCOPE, kind="collateral", args={"tickets": [tid]})]
        neg_sigs = [s["id"] for s in sigs.get(case, [])]
        trap = None
        if spec.get("mistake"):
            trap = f"the recorded first response to this request was later corrected ({', '.join(s['type'] for s in sigs.get(case, []))})"
        focus = f'Ticket {tid} ({spec["kind"]}): the customer wrote "{opener["text"]}"'
        instruction = " ".join([INTRO, focus, *[p.text for p in props]])
        tasks.append(TaskSpec(task_id=f"toy/helpdesk_hindsight/{case.lower()}", title=f"Hindsight {case}: {spec['kind']}", instruction=instruction,
                              properties=props, literals=[str(tid), oid], trap=trap, provenance=neg_sigs + [f"trace:toy/support_ticket:{case}"],
                              difficulty={"apps_used": 1, "traps": int(bool(trap))}))
        oracles[case] = None
        negatives[case] = spec.get("mistake")
    sc = {"seed": seed_ticket, "supervisor": SUPERVISOR, "customers": customers, "orders": orders,
          "distractors": {"X0": {"id": "O-6999", "customer": "A", "amount": 25.0, "status": "delivered"}},
          "tickets": tickets, "other_tickets": {"Y0": seed_ticket + 20}}
    return {"scenario": sc, "tasks": tasks, "mistakes": negatives, "signals": sigs, "known": known}


def env_files(b: dict[str, Any]) -> dict[str, str]:
    sc = b["scenario"]
    q = lambda s: "'" + str(s).replace("'", "''") + "'"
    schema = (PACK_ROOT / "data" / "tickets.sql").read_text().split("INSERT INTO")[0].strip()
    schema = schema.replace("CHECK (status IN ('open','pending','resolved','escalated'))", "CHECK (status IN ('open','pending','resolved','escalated'))")
    cust = ", ".join(f"({q(c['id'])}, {q(c['name'])}, {q(c['tier'])})" for c in sc["customers"].values())
    allo = {**sc["orders"], **sc["distractors"]}
    ords = ", ".join(f"({q(o['id'])}, {q(sc['customers'][o['customer']]['id'])}, {o['amount']:.2f}, {q(o['status'])})" for o in allo.values())
    logs = load_cases()
    trows = []
    for k, (case, tid) in enumerate(sc["tickets"].items()):
        opener = next(e for e in logs[case] if e["type"] == "ticket_opened")
        L = "ABCDEFGH"[k]
        trows.append(f"({tid}, {q(sc['customers'][L]['id'])}, {q(opener['text'][:80])}, 'open', '2026-04-01T08:{10 + k:02d}:00Z', NULL)")
    trows.append(f"({sc['other_tickets']['Y0']}, {q(sc['customers']['A']['id'])}, 'Update my billing address', 'open', '2026-04-01T08:40:00Z', NULL)")
    sql = "\n".join([schema, f"INSERT INTO customers VALUES {cust};", f"INSERT INTO orders VALUES {ords};", "INSERT INTO tickets VALUES " + ",\n  ".join(trows) + ";"]) + "\n"
    records = "".join(json.dumps({"id": f"hist#{c['id']}", "source": "account", "author": c["id"], "role": f"customer_{L.lower()}", "time": "2026-04-01T08:00:00Z",
                                  "text": f"I am {c['name']} ({c['id']})."}) + "\n" for L, c in sc["customers"].items())
    world = {"schema_version": "crucible.world/v1", "world": "toy/helpdesk_hindsight",
             "description": "Helpdesk world rebuilt from recorded ticket cases (G-hindsight): their requests, accounts and the policies corrections revealed.",
             "backend": "local", "backend_params": {"bundles": [{"src": "data/helpdesk", "dest": "helpdesk"}]},
             "apps": [{"id": "tickets", "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": "helpdesk/db/tickets.db"}},
                       "params": {"schema_sql": "data/tickets.sql", "tools_module": "helpdesk/tools/tickets.py", "transport": "inprocess"}},
                      {"id": "kb", "template": "common/doc_library", "state": {"adapter": "fs", "params": {"root": "kb"}}, "params": {"seed_dir": "data/kb"}}],
             "actors": [{"id": "agent", "kind": "policy", "tools": ["tickets.*", "kb.*"]}, {"id": "supervisor", "kind": "system"},
                        *({"id": f"customer_{L.lower()}", "kind": "simulated", "grounded_on": ["data/customers.jsonl"]} for L in sc["customers"])],
             "clock": {"mode": "step", "t0": T0, "step_dt_s": 300}}
    return {"data/tickets.sql": sql, "data/kb/refund_policy.md": CURRENT_POLICY, "data/kb/billing_corrections.md": EXTRA_POLICY,
            "data/customers.jsonl": records, "data/helpdesk/tools/tickets.py": (PACK_ROOT / "data" / "helpdesk" / "tools" / "tickets.py").read_text(),
            "worlds/helpdesk_hindsight.yaml": yaml.safe_dump(world, sort_keys=False)}


async def forge_hindsight(out: Path | None = None) -> dict[str, Any]:
    b = build()
    out = Path(out or CRUCIBLE / "data" / "forge" / "toy_hindsight")
    from crucible.forge.schema import EnvBuild

    write_pack(EnvBuild(world_id="toy/helpdesk_hindsight", world_yaml="", files=env_files(b), frozen_now=T0), out)
    sc = b["scenario"]
    signals = {"approval": next((s["id"] for ss in b["signals"].values() for s in ss if s.get("rule") == "refund_over_50_needs_approval"), "")}
    used_sig = next(s["id"] for s in b["signals"]["T1021"] if s["type"] == "revert")  # the correction that revealed the rule
    signals["forbidden:used"] = used_sig
    extra = [("refund_used_addon", "SELECT r.id, r.order_id FROM refunds r JOIN orders o ON o.id = r.order_id WHERE o.status = 'used'", used_sig)]
    reports, rows = [], {}
    for t in b["tasks"]:
        case = t.task_id.split("/")[-1].upper()
        oracle = oracle_actions(t, sc)
        checks = compile_checks(t, sc, signals, extra)
        negs = negatives_for(t, oracle, sc, signals)
        mistake = b["mistakes"].get(case)
        if mistake:  # the expert's recorded first response, replayed: it must fail
            tid = sc["tickets"][case]
            acts = [{**a, "arguments": {**a["arguments"], "ticket_id": tid}} for a in mistake]
            acts += [T(9, "tickets.add_note", ticket_id=tid, body="Handled."), T(10, "tickets.set_status", ticket_id=tid, status="resolved")] \
                if not any(a["function_name"] == "tickets.set_status" for a in acts) else [T(9, "tickets.add_note", ticket_id=tid, body="Handled.")]
            sig = next((s["id"] for s in b["signals"].get(case, []) if s["type"] in ("violation", "revert", "rework", "correction")), None)
            gate = "policy" if case in ("T1003", "T1015", "T1021") else "outcome"
            negs.append(NegativeSpec(id=f"{case.lower()}/recorded_first_response", task_id=t.task_id, kind="trap_reversion", actions=acts, expected_gate=gate,
                                     provenance=sig, description=f"the expert's recorded first response on {case}, later corrected"))
        task = ToyForgedTask(t, checks, oracle, negs, pack_root=out, world_file="worlds/helpdesk_hindsight.yaml", scenario=sc)
        r = await accept(task, resolve=lambda s: s in b["known"])
        reports.append(r)
        rows[case] = {"accepted": r.ok, "failing": [c.id for c in r.checks if c.status == "fail"], "trap": t.trap, "provenance": t.provenance,
                      "negatives": len(negs), "oracle": [a.get("function_name") or f"message:{a.get('to')}" for a in oracle],
                      "reasons": {c.id: c.detail.get("reasons") for c in r.checks if c.status == "fail"}}
    rep = {"gate": "M8 G-hindsight (toy): recorded ticket cases -> tasks graded on what held up", "summary": summarize(reports), "cases": rows,
           "pass": all(r.ok for r in reports), "note": "T1003's fee rule is inferred from a single supervisor correction (240.00 -> 230.00)"}
    return rep


def main() -> None:
    rep = asyncio.run(forge_hindsight())
    (CRUCIBLE / "docs" / "modules" / "reports" / "m8_toy_hindsight.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({"pass": rep["pass"], "summary": rep["summary"], "cases": {k: (v["accepted"], v["failing"], v["reasons"]) for k, v in rep["cases"].items()}},
                     indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
