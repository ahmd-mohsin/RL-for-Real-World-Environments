"""The backlog as a task: per-ticket subgoals (the grade), a ``RunTask`` for the M10 runner, and the oracle policy.

A ticket is correct iff, in the current state and the episode so far:
- its refunds are exactly the expected one (order and amount) or none;
- a refund needing approval under the rule in force when it was issued (threshold 50, then 25 after the amendment) has
  ``approved_by`` = the supervisor **and** an approval of that amount from the supervisor earlier in the episode; any
  recorded approver needs such an approval (``approval_grounded``, B.15);
- an awaiting ticket was acted on (refunded or resolved) only after the customer's reply fired;
- it has a note from the agent and the expected status.

reward = fraction of tickets correct = Φ at the end, so the dense progress rewards telescope to the reward exactly.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from crucible.core.schema.state import Query
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.verify.progress import Subgoal
from crucible.verify.runtime.plan import GradeReport
from crucible.world.world import World
from crucible_pack_toy.longhorizon.world import Backlog, draw, instruction, oracle, write_pack

APPROVAL = re.compile(r"[Aa]pproved refund of (\d+(?:\.\d+)?)")


class BacklogSubgoals:
    def __init__(self, b: Backlog):
        self.b = b
        self.tickets = {f"ticket:{t.id}": t for t in b.tickets}
        self.subgoals = [Subgoal(f"ticket:{t.id}", frozenset({f"ticket:{t.id}", f"order:{t.order}", *([f"event:{t.follow_up['id']}"] if t.follow_up else [])}))
                         for t in b.tickets]

    def _events(self, world: Any) -> tuple[list[tuple[str, float]], dict[str, str]]:
        approvals, fired = [], {}
        for e in world.log:
            if e.type == "message" and e.actor == "supervisor":
                approvals += [(e.time or "", float(x)) for x in APPROVAL.findall(e.text or "")]
            elif e.type == "event:nl" and (e.data or {}).get("event", "").startswith("followup_"):
                fired[e.data["event"]] = (e.time or "")
        return approvals, fired

    async def evaluate(self, world: Any, ids: Iterable[str]) -> dict[str, bool]:
        ids = [i for i in ids if i in self.tickets]
        if not ids:
            return {}
        tids = ",".join(str(self.tickets[i].id) for i in ids)
        res = (await world.verifier_view().query("tickets", Query(op="sql", args={"max_rows": 100000, "sql":
               f"SELECT t.id, t.status, t.resolved_at, (SELECT COUNT(*) FROM notes n WHERE n.ticket_id = t.id AND n.author = 'agent') AS notes, "
               f"(SELECT json_group_array(json_array(r.order_id, r.amount_usd, r.approved_by, r.created_at)) FROM refunds r WHERE r.ticket_id = t.id) AS refunds "
               f"FROM tickets t WHERE t.id IN ({tids})"}))).value or {}
        if res.get("truncated"):
            raise RuntimeError("subgoal query truncated")
        rows = res.get("rows") or []
        approvals, fired = self._events(world)
        out = {}
        for r in rows:
            rid, status, resolved_at, notes, refunds = r
            t = self.tickets[f"ticket:{rid}"]
            out[f"ticket:{rid}"] = self._correct(t, status, resolved_at, notes, json.loads(refunds or "[]"), approvals, fired)
        return out

    @staticmethod
    def _norm(ts: str) -> str:
        return (ts or "").replace("Z", "").split("+")[0][:19]

    def _correct(self, t, status, resolved_at, notes, refunds, approvals, fired) -> bool:
        exp = t.expected
        if status != exp["status"] or notes < 1:
            return False
        if exp.get("wait_for"):
            at = fired.get(exp["wait_for"])
            if at is None or self._norm(resolved_at) < self._norm(at):
                return False
            if any(self._norm(r[3]) < self._norm(at) for r in refunds):
                return False
        if exp["refund"] is None:
            return not refunds
        if len(refunds) != 1:
            return False
        order, amount, approver, created = refunds[0]
        if order != t.order or abs(amount - exp["refund"]) > 0.005:
            return False
        needs = amount > self.b.threshold_at(self._norm(created))
        if needs and approver != self.b.supervisor:
            return False
        # recorded approvals must have been given in the episode, for that amount, before the refund
        return not approver or any(self._norm(at) <= self._norm(created) and abs(a - amount) < 0.005 for at, a in approvals)


class BacklogTask:
    """``RunTask`` for the M10 runner (``open`` a fresh started World, ``grade`` it)."""

    def __init__(self, seed: int = 0, n: int = 40, *, root: Path | None = None):
        self.b = draw(seed, n)
        self.root = Path(root or tempfile.mkdtemp(prefix="backlog-pack-"))
        write_pack(self.b, self.root)
        self.id = f"toy/helpdesk_backlog/s{seed}-n{n}@1"
        self.world_id = "toy/helpdesk_backlog"
        self.instruction = instruction(self.b)
        self.subgoals = BacklogSubgoals(self.b)

    def world(self, workdir: Path) -> World:
        from crucible.actors.scripted import RelayActor, Rule

        w = World(WorldSpec.load(self.root / "worlds" / "helpdesk_backlog.yaml"), instructions={"agent": self.instruction}, store=Store(Path(workdir) / "s"),
                  workdir=Path(workdir) / "w", pack_root=self.root)
        w.attach(RelayActor("supervisor", [Rule(r"approv\w*.*?(\d+(?:\.\d+)?)\s*usd", reply=f"Approved refund of {{0}} USD. - {self.b.supervisor}")],
                            default="Please state the ticket, the order and the amount you want approved."))
        return w

    async def open(self, workdir: Path) -> World:
        w = self.world(workdir)
        await w.start()
        await w.reset()
        return w

    async def grade(self, world: World, workdir: Path) -> GradeReport:
        ids = [g.id for g in self.subgoals.subgoals]
        res = await self.subgoals.evaluate(world, ids)
        frac = sum(res.values()) / len(ids)
        wrong = sorted(k for k, v in res.items() if not v)
        by_kind: dict[str, list[int]] = {}
        for k, v in res.items():
            by_kind.setdefault(self.subgoals.tickets[k].kind, [0, 0])
            by_kind[self.subgoals.tickets[k].kind][0] += int(v)
            by_kind[self.subgoals.tickets[k].kind][1] += 1
        return GradeReport(task=self.id, tier="full", checks=[], gates={}, reward=round(frac, 6), outcome=round(frac, 6), infra={"status": "ok"},
                           trusted=True, elapsed_s=0.0, meta={"correct": sum(res.values()), "tickets": len(ids), "wrong": wrong[:50], "by_kind": by_kind})

    def oracle_actions(self) -> list[dict]:
        return oracle(self.b)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def oracle_policy(task: BacklogTask):
    from crucible.actors.policy import ScriptedPolicy
    from crucible.core.schema.actions import Message, ToolCall
    from crucible.runtime.harness.native import PolicyRef

    acts = [ToolCall(tool_call_id=a["tool_call_id"], function_name=a["function_name"], arguments=a.get("arguments") or {}) if a["kind"] == "tool_call"
            else Message(to=a["to"], text=a["text"]) for a in task.oracle_actions()]
    return PolicyRef(id="scripted:backlog-oracle", make=lambda attempt: {"agent": ScriptedPolicy("agent", list(acts))})
