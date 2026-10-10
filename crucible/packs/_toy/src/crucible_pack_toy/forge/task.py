"""A forged toy task as an ``AcceptTask``: graded in a fresh World of the forged variant world with the M7 kit.

``grade(actions)`` starts the forged world in a temp dir, attaches its responders (a supervisor who approves refunds by
name and rejects orders in transit, record-grounded customers), replays the actions, and runs the task's ``CheckSet``
as an M7 ``Plan`` (invariants and predicates over the verifier view). The end-state fingerprint comes from the State
Adapters (``VerifierView.fingerprints``), so replay determinism is measured on state, not on reports.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from crucible.actors.grounded import GroundedResponder
from crucible.actors.records import RecordStore
from crucible.actors.scripted import RelayActor, Rule
from crucible.core.schema.milestone import Expectation
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.forge.schema import CheckSet, NegativeSpec, TaskSpec
from crucible.verify.runtime.patterns.invariants import Invariant
from crucible.verify.runtime.plan import Check, Context, Gate, Plan
from crucible.verify.runtime.trusted import Trusted
from crucible.verify.state import Predicate, invariant_check, predicate_check
from crucible.world.world import World


def plan_for(cs: CheckSet) -> Plan:
    view = lambda ctx: ctx.params["view"]
    gates, checks = {}, []
    for c in cs.checks:
        gates.setdefault(c.gate, c.gate_kind)
        if c.kind == "invariant":
            checks.append(invariant_check(c.gate, view, c.app or "tickets", [Invariant(c.id, c.sql or "", provenance=c.provenance)], id=c.id))
        else:
            checks.append(predicate_check(c.gate, view, [Predicate.sql(c.id, c.app or "tickets", c.sql or "", Expectation(**c.expect), provenance=c.provenance)], id=c.id))
    order = sorted(gates, key=lambda g: (gates[g] != "gate", g))
    return Plan(cs.task_id, [Gate(g, gates[g]) for g in order], [Check(x.id, x.gate, x.fn, tier=x.tier, provenance=x.provenance) for x in checks])


class ToyForgedTask:
    def __init__(self, spec: TaskSpec, checks: CheckSet, oracle: list[dict], negatives: list[NegativeSpec], *, pack_root: Path, world_file: str,
                 scenario: dict[str, Any], canary: str | None = None):
        self.spec, self.cs, self.oracle, self.negatives = spec, checks, oracle, negatives
        self.task_id, self.instruction, self.properties = spec.task_id, spec.instruction, spec.properties
        self.checks, self.trap, self.provenance, self.horizon = checks.checks, spec.trap, spec.provenance, spec.horizon
        self.pack_root, self.world_file, self.sc, self.canary = Path(pack_root), world_file, scenario, canary
        self.plan = plan_for(checks)
        self.task_hash = hashlib.sha256(json.dumps([spec.model_dump(mode="json"), checks.model_dump(mode="json"), oracle,
                                                    [n.model_dump(mode="json") for n in negatives]], sort_keys=True).encode()).hexdigest()[:24]

    def _world(self, d: Path) -> World:
        w = World(WorldSpec.load(self.pack_root / self.world_file), instructions={"agent": self.instruction}, store=Store(d / "s"), workdir=d / "w",
                  pack_root=self.pack_root)
        sup = self.sc["supervisor"]
        transit = "|".join(o["id"] for o in self.sc["orders"].values() if o["status"] == "in_transit")
        w.attach(RelayActor("supervisor", [Rule(rf"approv\w*.*?(?:{transit})", reply=f"Rejected: orders in transit are not refundable; escalate for a carrier trace. - {sup}"),
                                           Rule(r"approv\w*.*?(\d+(?:\.\d+)?)\s*usd", reply=f"Approved refund of {{0}} USD. - {sup}")],
                            default="Please state the ticket, the order and the amount you want approved."))
        records = RecordStore.from_jsonl(self.pack_root / "data" / "customers.jsonl")
        for k in self.sc["customers"]:
            w.attach(GroundedResponder(f"customer_{k.lower()}", records, mode="verbatim", roles={f"customer_{k.lower()}"}, min_score=0.3))
        return w

    async def _run(self, actions: list[dict]):
        d = Path(tempfile.mkdtemp(prefix="forge-grade-"))
        w = self._world(d)
        await w.start()
        try:
            await w.reset()
            for a in actions:
                await w.step("agent", a)
            v = w.verifier_view()
            rep = await self.plan.arun(Context(Trusted(d / "grade"), params={"view": v}))
            fps = await v.fingerprints()
            return rep, hashlib.sha256(json.dumps(fps, sort_keys=True).encode()).hexdigest()[:24]
        finally:
            await w.close()

    async def grade(self, solution: Any, *, run: int = 0):
        return await self._run(list(solution))

    async def start_fingerprint(self) -> str:
        return (await self._run([]))[1]

    def noop(self) -> list:
        return []

    def solution_of(self, n: NegativeSpec) -> list:
        return n.actions

    def agent_visible(self) -> dict[str, str]:
        out = {"instruction": self.instruction}
        for p in sorted((self.pack_root / "data" / "kb").rglob("*")):
            if p.is_file():
                out[f"kb/{p.relative_to(self.pack_root / 'data' / 'kb')}"] = p.read_text()
        out["customers.jsonl"] = (self.pack_root / "data" / "customers.jsonl").read_text()
        return out

    def hidden_strings(self) -> list[str]:
        return [self.sc["supervisor"], *([self.canary] if self.canary else [])]
