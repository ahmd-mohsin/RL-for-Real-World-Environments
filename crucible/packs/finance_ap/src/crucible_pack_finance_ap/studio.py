"""Studio hooks of the finance_ap pack (declared in ``pack.yaml``): the AP invoice World as an M10 ``RunTask`` with Φ per
step (one subgoal per invoice), gold and designed negatives, a seeded generator, and scripted policies.

Grade (state only, from the ERP database):
- **Gate `controls`.** No invoice that a policy rule should have stopped was approved. Paying a duplicate, a blocked
  vendor, unverified bank details, an over-tolerance price or unreceived goods is a control failure, and the episode
  scores 0 however many other invoices were right.
- **Outcome.** The share of invoices in their correct final state with the policy's reason code.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from crucible.core.schema.state import Query
from crucible.core.store import Store
from crucible.verify.progress import Subgoal
from crucible.verify.runtime.plan import GradeReport
from crucible.world.world import World
from crucible_pack_finance_ap.world import EXPECT, draw, instruction, oracle, write_pack

_MADE: dict[str, Any] = {}


class InvoiceSubgoals:
    def __init__(self, w):
        self.cases = {f"invoice:{c.invoice_id}": c for c in w.cases}
        self.subgoals = [Subgoal(k, frozenset({k, f"vendor:{c.vendor_id}", f"po:{c.po_id}"})) for k, c in self.cases.items()]

    async def state(self, world: Any) -> dict[int, tuple[str, str | None]]:
        res = (await world.verifier_view().query("erp", Query(op="sql", args={"sql": "SELECT id, status, reason FROM invoices", "max_rows": 100000}))).value or {}
        return {int(r[0]): (r[1], r[2]) for r in res.get("rows") or []}

    async def evaluate(self, world: Any, ids: Iterable[str]) -> dict[str, bool]:
        st = await self.state(world)
        return {i: st.get(self.cases[i].invoice_id) == self.cases[i].expected for i in ids if i in self.cases}


class APTask:
    def __init__(self, seed: int = 0, n: int = 7):
        self.w = draw(seed, n)
        self.root = Path(tempfile.mkdtemp(prefix="finance-ap-"))
        write_pack(self.w, self.root)
        self.id = f"finance_ap/invoices/s{seed}-n{n}@1"
        self.world_id = "finance_ap/invoices"
        self.instruction = instruction(self.w)
        self.subgoals = InvoiceSubgoals(self.w)
        self.oracle = oracle(self.w)

    async def open(self, workdir: Path) -> World:
        from crucible.core.schema.world import WorldSpec

        w = World(WorldSpec.load(self.root / "worlds" / "ap.yaml"), instructions={"agent": self.instruction}, store=Store(Path(workdir) / "s"),
                  workdir=Path(workdir) / "w", pack_root=self.root)
        await w.start()
        await w.reset()
        return w

    async def grade(self, world: World, workdir: Path) -> GradeReport:
        st = await self.subgoals.state(world)
        correct = {c.invoice_id: st.get(c.invoice_id) == c.expected for c in self.w.cases}
        breaches = [c.invoice_id for c in self.w.cases if c.expected[0] != "approved" and (st.get(c.invoice_id) or ("", None))[0] == "approved"]
        frac = sum(correct.values()) / len(correct)
        gate = "fail" if breaches else "pass"
        return GradeReport(task=self.id, tier="full", checks=[], gates={"controls": gate}, reward=0.0 if breaches else round(frac, 6),
                           outcome=round(frac, 6), infra={"status": "ok"}, trusted=True, elapsed_s=0.0,
                           meta={"correct": sum(correct.values()), "invoices": len(correct), "control_breaches": breaches,
                                 "wrong": sorted(k for k, v in correct.items() if not v), "tolerance": self.w.tolerance})

    def oracle_actions(self) -> list[dict]:
        return list(self.oracle)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _acts(raw: list[dict]) -> list:
    from crucible.core.schema.actions import Message, ToolCall

    return [ToolCall(tool_call_id=a["tool_call_id"], function_name=a["function_name"], arguments=a.get("arguments") or {}) if a["kind"] == "tool_call"
            else Message(to=a["to"], text=a["text"]) for a in raw]


def tasks() -> list:
    return [(t, _acts(t.oracle)) for t in (APTask(1), APTask(2))]


def generate(seed: int) -> APTask:
    t = APTask(seed % 100_000)
    _MADE[t.id] = t
    return t


def stages(task_id: str):
    t = _MADE.get(task_id)
    if t is None:
        return None
    ids = [g.id for g in t.subgoals.subgoals]
    return [(f"q{q}", ids[: max(1, len(ids) * q // 4)]) for q in (1, 2, 3, 4)]


def policies() -> list:
    from crucible.actors.policy import ScriptedPolicy
    from crucible.runtime.harness.native import PolicyRef

    def approve_all(task):  # the classic AP failure: pay everything that arrives
        return [{"kind": "tool_call", "tool_call_id": f"a{c.invoice_id}", "function_name": "erp.approve_invoice", "arguments": {"invoice_id": c.invoice_id}}
                for c in task.w.cases]

    return [PolicyRef(id="scripted:oracle", make=lambda a: {"agent": ScriptedPolicy("agent", _acts(a.task.oracle))}),
            PolicyRef(id="scripted:approve_all", make=lambda a: {"agent": ScriptedPolicy("agent", _acts(approve_all(a.task)))}),
            PolicyRef(id="scripted:noop", make=lambda a: {"agent": ScriptedPolicy("agent", [])})]


def fixtures() -> list[dict]:
    """Gold, and designed negatives that each break one control or one decision."""
    t = APTask(3)
    gold = list(t.oracle)

    def decide(inv: int, fn: str, **kw) -> list[dict]:
        out = []
        for a in gold:
            if a.get("arguments", {}).get("invoice_id") == inv and a["function_name"] in ("erp.approve_invoice", "erp.hold_invoice", "erp.reject_invoice"):
                out.append({**a, "function_name": fn, "arguments": {"invoice_id": inv, **kw}})
            else:
                out.append(a)
        return out

    by_kind = {c.kind: c for c in t.w.cases}
    n = len(t.w.cases)
    cases = [{"name": "gold", "task": t, "actions": _acts(gold), "reward": 1.0}]
    for kind in ("duplicate", "vendor_blocked", "bank_change", "price_variance", "missing_receipt"):
        c = by_kind[kind]
        cases.append({"name": f"approves_{kind}", "task": t, "actions": _acts(decide(c.invoice_id, "erp.approve_invoice")), "reward": 0.0, "fails_gate": "controls"})
    c = by_kind["within_tolerance"]
    cases.append({"name": "holds_within_tolerance", "task": t, "actions": _acts(decide(c.invoice_id, "erp.hold_invoice", reason="price_variance")),
                  "reward": round((n - 1) / n, 6)})
    c = by_kind["price_variance"]
    cases.append({"name": "wrong_reason_code", "task": t, "actions": _acts(decide(c.invoice_id, "erp.hold_invoice", reason="missing_receipt")),
                  "reward": round((n - 1) / n, 6)})
    cases.append({"name": "noop", "task": t, "actions": [], "reward": 0.0})
    return cases


EXPECTED_KINDS = sorted(EXPECT)
