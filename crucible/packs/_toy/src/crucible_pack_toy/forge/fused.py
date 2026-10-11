"""A fused toy task (m13 G-9): a forged ``toy/helpdesk_v2`` task ⊕ a warehouse task, in one World with both Apps.

The warehouse World (``inventory``: shipped order items, stock, returns) is generated from the helpdesk scenario's
orders. The warehouse task: log the return of every refunded order's items as damaged (quarantined, never restocked).
The two components touch different Apps and are independent (A17 applies); the cross-App check ``x.consistency`` is
the interaction neither part has: the refunded orders in the ticketing system and the returned orders in the
warehouse must be the same set (a refund without its return, or a return without its refund, fails it).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from crucible.core.schema.world import WorldSpec
from crucible.forge.compose import Parts, fuse
from crucible.forge.fuse import fuse_worlds
from crucible.forge.schema import CheckSet, CheckSpec, NegativeSpec, StatedProperty, TaskSpec
from crucible_pack_toy import PACK_ROOT
from crucible_pack_toy.forge.task import ToyForgedTask

STOCK = 5


def _sku(order_id: str) -> str:
    return f"SKU-{order_id.split('-', 1)[-1]}"


def warehouse_world(sc: dict[str, Any]) -> tuple[WorldSpec, str]:
    rows = [f"('{o['id']}', '{_sku(o['id'])}', 1)" for o in sc["orders"].values()]
    stock = [f"('{_sku(o['id'])}', {STOCK})" for o in sc["orders"].values()]
    sql = ("CREATE TABLE order_items (order_id TEXT NOT NULL, sku TEXT NOT NULL, qty INTEGER NOT NULL, PRIMARY KEY (order_id, sku));\n"
           "CREATE TABLE stock (sku TEXT PRIMARY KEY, qty INTEGER NOT NULL);\n"
           "CREATE TABLE returns (id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT NOT NULL, sku TEXT NOT NULL, qty INTEGER NOT NULL, "
           "condition TEXT NOT NULL CHECK (condition IN ('resellable','damaged')));\n"
           f"INSERT INTO order_items VALUES {', '.join(rows)};\nINSERT INTO stock VALUES {', '.join(stock)};\n")
    w = WorldSpec(world="toy/warehouse", description="A small warehouse: shipped items, stock, returns.", backend="local",
                  backend_params={"bundles": [{"src": "data/warehouse", "dest": "warehouse"}]},
                  apps=[{"id": "inventory", "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": "warehouse/db/inventory.db"}},
                         "params": {"schema_sql": "data/inventory.sql", "tools_module": "warehouse/tools/inventory.py", "transport": "inprocess"},
                         "description": "warehouse inventory and returns"}],
                  actors=[{"id": "agent", "kind": "policy", "tools": ["inventory.*"]}])
    return w, sql


def warehouse_parts(helpdesk: ToyForgedTask, task_id: str) -> Parts:
    orders = sorted({a["arguments"]["order_id"] for a in helpdesk.oracle if a.get("function_name") == "tickets.issue_refund"})
    if not orders:
        raise ValueError(f"{helpdesk.task_id} refunds nothing: no returns to log")
    q = ", ".join(f"'{o}'" for o in orders)
    text = (f"The customer sent back the items of order(s) {', '.join(orders)}, damaged. Log each return in the warehouse as damaged; "
            "damaged items go to quarantine and are never put back in stock.")
    props = [StatedProperty(id="returns", text=text.split(". ")[1], kind="outcome", args={"orders": orders}),
             StatedProperty(id="no_restock", text="damaged items go to quarantine and are never put back in stock", kind="policy")]
    spec = TaskSpec(task_id=task_id, title=f"log damaged returns {', '.join(orders)}", instruction=text, properties=props, literals=orders,
                    provenance=["kb:warehouse_returns (damaged -> quarantine)"])
    checks = CheckSet(task_id=task_id, checks=[
        CheckSpec(id="returns/logged_damaged", gate="returns", kind="predicate", app="inventory",
                  sql=f"SELECT COUNT(DISTINCT order_id) FROM returns WHERE order_id IN ({q}) AND condition = 'damaged'",
                  expect={"kind": "between", "lo": len(orders), "hi": len(orders)}, states=["returns"]),
        CheckSpec(id="policy/damaged_not_restocked", gate="policy", gate_kind="gate", kind="invariant", app="inventory",
                  sql=f"SELECT sku, qty FROM stock WHERE qty != {STOCK}", states=["no_restock"]),
        CheckSpec(id="scope/other_returns", gate="scope", gate_kind="gate", kind="invariant", app="inventory",
                  sql=f"SELECT id, order_id FROM returns WHERE order_id NOT IN ({q})", states=["returns"])])
    oracle = [{"kind": "tool_call", "tool_call_id": f"wh{i}", "function_name": "inventory.log_return",
               "arguments": {"order_id": o, "sku": _sku(o), "qty": 1, "condition": "damaged"}} for i, o in enumerate(orders)]
    negs = [NegativeSpec(id="naive", task_id=task_id, kind="floor", actions=[], expected_gate="returns"),
            NegativeSpec(id="restocked_damaged", task_id=task_id, kind="trap_reversion", expected_gate="policy",
                         actions=[{**a, "arguments": {**a["arguments"], "condition": "resellable"}} for a in oracle],
                         description="logged as resellable: the damaged items go back on the shelf")]
    return Parts(spec, checks, oracle, negs)


CROSS = CheckSpec(id="consistency/refunds_match_returns", gate="consistency", gate_kind="gate", kind="cross",
                  spec={"left": {"app": "tickets", "sql": "SELECT DISTINCT order_id FROM refunds"},
                        "right": {"app": "inventory", "sql": "SELECT DISTINCT order_id FROM returns"}, "relation": "equal"},
                  provenance="fuse: a refund is matched by its return (cross-App)")


class ToyFusedTask(ToyForgedTask):
    def __init__(self, composite, sources: list[Any], *, pack_root: Path, world_file: str, scenario: dict[str, Any]):
        super().__init__(composite.spec, composite.checks, composite.oracle, composite.negatives, pack_root=pack_root, world_file=world_file, scenario=scenario)
        self.composite, self._sources = composite, list(sources)

    def component_tasks(self) -> list[Any]:
        return list(self._sources)


def build_fused(helpdesk: ToyForgedTask, out: Path) -> ToyFusedTask:
    """Writes the fused pack root (the forged helpdesk root + the warehouse App) and returns the fused task."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    for sub in ("data", "worlds"):
        shutil.copytree(helpdesk.pack_root / sub, out / sub)
    shutil.copytree(PACK_ROOT / "data" / "warehouse" / "tools", out / "data" / "warehouse" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    ww, sql = warehouse_world(helpdesk.sc)
    (out / "data" / "inventory.sql").write_text(sql)
    (out / "worlds" / "warehouse.yaml").write_text(yaml.safe_dump(ww.model_dump(mode="json", exclude_none=True), sort_keys=False))
    hw = WorldSpec.load(helpdesk.pack_root / helpdesk.world_file)
    fw, renames = fuse_worlds(hw, ww, world="toy/helpdesk_v2+warehouse")
    wf = "worlds/helpdesk_v2_warehouse.yaml"
    (out / wf).write_text(yaml.safe_dump(fw.model_dump(mode="json", exclude_none=True), sort_keys=False))
    tid = f"{helpdesk.task_id}+warehouse"
    wh = warehouse_parts(helpdesk, f"toy/warehouse/returns_for_{helpdesk.task_id.split('/')[-1]}")
    comp = fuse(Parts.of(helpdesk), wh, task_id=tid, cross=[CROSS], renames=renames)
    return ToyFusedTask(comp, [helpdesk], pack_root=out, world_file=wf, scenario=helpdesk.sc)
