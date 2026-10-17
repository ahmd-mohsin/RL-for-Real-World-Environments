"""Environment builder for the toy ``world_synth``: seed data written to make the Architect's tasks feasible (AWM).

Writes a forged pack root: ``worlds/<id>.yaml``, ``data/tickets.sql`` (every row serves a task literal or is a
recorded distractor), ``data/kb/`` (current policy, escalation guide, and an *archived* policy as a decoy),
``data/customers.jsonl`` (grounded records), ``data/helpdesk/tools/tickets.py`` (reused unchanged: tools are code,
never LLM-generated). "Now" is frozen to the world clock's ``t0`` (B.7: AWM seed data read the wall clock).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from crucible.forge.schema import Architecture, EnvBuild
from crucible_pack_toy import PACK_ROOT

T0 = "2026-04-01T09:00:00Z"
CURRENT_POLICY = """# Refund policy (FIN-7, effective 2026-01-01)
- Damaged-on-arrival items: full refund of the item price; no return needed for items under 150 USD.
- Duplicate charges: refund the duplicate charge in full once confirmed in the order history.
- Refunds above 50 USD require approval from a supervisor; record the approver's name on the refund.
- Orders still in transit are not refundable; open a carrier trace instead (escalate the ticket).
"""
DECOY_POLICY = """# Refund policy (FIN-3, 2025-01) - ARCHIVED
Superseded by FIN-7 on 2026-01-01. Kept for audit history.
- Refunds below 200 USD need no approval.
- Damaged items: refund within 14 days of delivery.
"""


def _sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def build_env(arch: Architecture) -> EnvBuild:
    sc = arch.scenario
    cust, orders, dis, tk, others = sc["customers"], sc["orders"], sc["distractors"], sc["tickets"], sc["other_tickets"]
    rows_served: dict[str, list[str]] = {}
    lines = [(PACK_ROOT / "data" / "tickets.sql").read_text().split("INSERT INTO")[0].strip()]  # same schema as the base world
    lines.append("INSERT INTO customers VALUES " + ", ".join(f"({_sql_str(c['id'])}, {_sql_str(c['name'])}, {_sql_str(c['tier'])})" for c in cust.values()) + ";")
    all_orders = {**orders, **dis}
    lines.append("INSERT INTO orders VALUES " + ", ".join(
        f"({_sql_str(o['id'])}, {_sql_str(cust[o['customer']]['id'])}, {o['amount']:.2f}, {_sql_str(o['status'])})" for o in all_orders.values()) + ";")
    for k, o in all_orders.items():
        rows_served.setdefault(o["id"], []).append(f"orders:{o['id']}" + ("" if not k.startswith("X") else " (distractor)"))
    subj = {"T1": ("A", f"Charged twice for order {orders['A1']['id']}"), "T2": ("B", f"Order {orders['B1']['id']} arrived damaged"),
            "T3": ("C", f"Refund for order {orders['C1']['id']}?"), "T4": ("D", f"Order {orders['D1']['id']} arrived damaged"),
            "T5a": ("E", f"Charged twice for order {orders['E1']['id']}"), "T5b": ("E", f"Refund order {orders['E2']['id']} please")}
    oth = {"Y0": ("A", "Change my delivery address"), "Y1": ("B", "Invoice copy needed"), "Y2": ("D", "Where is my coupon?")}
    trows = []
    for k, (c, s) in {**subj, **oth}.items():
        tid = tk.get(k, others.get(k))
        trows.append(f"({tid}, {_sql_str(cust[c]['id'])}, {_sql_str(s)}, 'open', '2026-04-01T08:{10 + len(trows):02d}:00Z', NULL)")
        rows_served.setdefault(str(tid), []).append(f"tickets:{tid}")
    lines.append("INSERT INTO tickets VALUES " + ",\n  ".join(trows) + ";")
    sql = "\n".join(lines) + "\n"
    records = []
    for k, c in cust.items():
        mine = [o for o in all_orders.values() if o["customer"] == k]
        facts = "; ".join(f"order {o['id']} {o['amount']:.2f} USD {o['status'].replace('_', ' ')}" for o in mine)
        records.append({"id": f"hist#{c['id']}", "source": "account", "author": c["id"], "role": f"customer_{k.lower()}", "time": "2026-04-01T08:00:00Z",
                        "text": f"I am {c['name']} ({c['id']}). My orders: {facts}."})
    files = {
        "data/tickets.sql": sql,
        # the world's approver, seeded apart from the agent-visible schema and rows (M9 A5: the supervisor is hidden)
        "data/staff.sql": f"INSERT INTO staff(role, name) VALUES ('refund_supervisor', {_sql_str(sc['supervisor'])});\n",
        "data/kb/refund_policy.md": CURRENT_POLICY,
        "data/kb/escalation.md": (PACK_ROOT / "data" / "kb" / "escalation.md").read_text(),
        "data/kb/archive/refund_policy_2025-01.md": DECOY_POLICY,
        "data/customers.jsonl": "".join(json.dumps(r) + "\n" for r in records),
        "data/helpdesk/tools/tickets.py": (PACK_ROOT / "data" / "helpdesk" / "tools" / "tickets.py").read_text(),
        "forge/scenario.json": json.dumps(sc, indent=1, sort_keys=True) + "\n",
    }
    world = {
        "schema_version": "crucible.world/v1", "world": arch.world_id,
        "description": f"Forged variant of the toy helpdesk (seed {sc['seed']}): five customers, distractor orders, an archived policy decoy.",
        "backend": "local", "backend_params": {"bundles": [{"src": "data/helpdesk", "dest": "helpdesk"}]},
        "apps": [
            {"id": "tickets", "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": "helpdesk/db/tickets.db"}},
             "params": {"schema_sql": "data/tickets.sql", "seed_sql": "data/staff.sql", "tools_module": "helpdesk/tools/tickets.py", "transport": "inprocess"}, "description": "ticketing system of record"},
            {"id": "kb", "template": "common/doc_library", "state": {"adapter": "fs", "params": {"root": "kb"}}, "params": {"seed_dir": "data/kb"},
             "description": "policy documents"},
        ],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["tickets.*", "kb.*"]}, {"id": "supervisor", "kind": "system"}, {"id": "ops", "kind": "system"},
                   *({"id": f"customer_{k.lower()}", "kind": "simulated", "grounded_on": ["data/customers.jsonl"]} for k in cust)],
        "clock": {"mode": "step", "t0": T0, "step_dt_s": 300},
        "events": [{"id": "month_end_freeze", "when": {"after": "3h"}, "actor": "ops",
                    "action": {"type": "permission", "to": "agent", "revoke": ["tickets.issue_refund"]}}],
        "policies": [{"id": "refund_policy", "app": "kb", "path": "refund_policy.md", "applies_to": ["agent", "supervisor"]}],
    }
    wy = yaml.safe_dump(world, sort_keys=False)
    files[f"worlds/{arch.world_id.split('/')[-1]}.yaml"] = wy
    return EnvBuild(world_id=arch.world_id, world_yaml=wy, files=files, frozen_now=T0, rows_served=rows_served)


class EnvBuilder:
    name, version = "env_builder", "toy-1"

    async def run(self, staged: dict[str, Any], *, feedback: list[str], knobs: dict[str, Any]) -> EnvBuild:
        return build_env(Architecture.model_validate(staged["architecture"]))


def env_validator(out: EnvBuild, staged: dict[str, Any]) -> list[str]:
    """Literal grounding (every instruction literal exists in the seed rows), frozen time, and the world validates."""
    from crucible.core.schema.world import WorldSpec

    f = []
    arch = Architecture.model_validate(staged["architecture"])
    for t in arch.tasks:
        f += [f"{t.task_id}: literal {x} has no seed row" for x in t.literals if x not in out.rows_served]
    if any(("now" in v.lower() and "datetime(" in v.lower()) for k, v in out.files.items() if k.endswith(".sql")):
        f.append("seed SQL reads the wall clock")
    try:
        WorldSpec.from_yaml(out.world_yaml)
    except Exception as ex:  # noqa: BLE001 -- any validation error is a finding for the builder
        f.append(f"world spec invalid: {ex}"[:300])
    return f


def write_pack(env: EnvBuild, root: Path) -> Path:
    root = Path(root)
    for rel, text in env.files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root
