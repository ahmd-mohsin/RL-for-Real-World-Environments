"""State predicates and invariants on live Worlds and reconstructed states (host side; DESIGN §8.1, P9, P12).

``state_predicate``: a query against one App plus an expectation. It is the milestone check format
(``core/schema/milestone.py``) and its value comes from the same ``workflow.milestones.evaluate.phi``, so a hidden check
and a milestone over the same property can never disagree. A check that cannot run is **infra**, not a fail
(a failed measurement is not a zero).

``view_query(view, app)`` adapts a ``VerifierView`` (or any state view) to the ``sql -> rows`` callable that
``runtime.patterns.invariants`` expects.

Built on: EnterpriseOps-Gym ``database_state`` / ``response_check`` verifiers (expected-row and response
predicates), turned into end-state predicates plus violating-row invariants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crucible.core.schema.milestone import Expectation, Milestone, MilestoneCheck
from crucible.core.schema.state import Query
from crucible.verify.runtime.patterns.invariants import Invariant, check_invariants
from crucible.verify.runtime.plan import Check, Verdict
from crucible.verify.runtime.trusted import InfraError
from crucible.workflow.milestones.evaluate import decode, phi


@dataclass
class Predicate:
    id: str
    app: str
    query: Query
    expect: Expectation = field(default_factory=Expectation)
    description: str = ""
    provenance: str | None = None

    @classmethod
    def sql(cls, id: str, app: str, sql: str, expect: Expectation, **kw: Any) -> Predicate:
        return cls(id=id, app=app, query=Query(op="scalar", args={"sql": sql}), expect=expect, **kw)

    def milestone(self, **kw: Any) -> Milestone:
        """The same property as a milestone (for Φ), sharing the check."""
        return Milestone(id=self.id, check=MilestoneCheck(app=self.app, query=self.query, expect=self.expect), **kw)


async def state_predicates(view, predicates: list[Predicate]) -> Verdict:
    out, unmeasured = {}, []
    for p in predicates:
        r = await view.query(p.app, p.query)
        v = phi(p.expect, r)
        if v is None:
            unmeasured.append(p.id)
            out[p.id] = {"measured": False, "error": (r.error or "")[:300]}
        else:
            out[p.id] = {"pass": v == 1.0, "value": decode(r.value), "description": p.description}
    if unmeasured:
        return Verdict(False, {"predicates": out}, infra=f"unmeasurable predicates: {unmeasured}")
    return Verdict(all(x["pass"] for x in out.values()), {"predicates": out})


def view_query(view, app: str):
    async def q(sql: str) -> list:
        r = await view.query(app, Query(op="sql", args={"sql": sql}))
        if not r.ok:
            raise InfraError(f"invariant query failed on {app}: {r.error}")  # the grader's query, never the candidate's fault
        v = decode(r.value)
        return v.get("rows", []) if isinstance(v, dict) else list(v or [])

    return q


def predicate_check(gate: str, view_of, predicates: list[Predicate], *, id: str | None = None, tier: str = "fast", requires: tuple = ()) -> Check:
    """A plan check over a state view; ``view_of(ctx)`` returns the view (e.g. ``ctx.params["view"]``)."""
    return Check(id=id or f"{gate}/predicates", gate=gate, fn=lambda ctx: state_predicates(view_of(ctx), predicates), tier=tier, requires=requires,
                 provenance=";".join(p.provenance for p in predicates if p.provenance) or None)


def invariant_check(gate: str, view_of, app: str, invariants: list[Invariant], *, id: str | None = None, tier: str = "fast") -> Check:
    return Check(id=id or f"{gate}/invariants", gate=gate, fn=lambda ctx: check_invariants(view_query(view_of(ctx), app), invariants), tier=tier,
                 provenance=";".join(i.provenance for i in invariants if i.provenance) or None)
