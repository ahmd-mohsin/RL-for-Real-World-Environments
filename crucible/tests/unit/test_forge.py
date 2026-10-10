"""M8 forge core: deterministic DAG, journal resume, retries with feedback, budgets, asymmetric information, learnability, dedup."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from crucible.core.store import Store
from crucible.forge.budget import BudgetExhausted, Ledger
from crucible.forge.orchestrator import Journal, Node, Orchestrator, Plan, RoleInfraError


class Doc(BaseModel):
    value: Any = None
    note: str = ""


class Echo:
    name, version = "echo", "1"

    def __init__(self):
        self.calls = 0

    async def run(self, staged, *, feedback, knobs):
        self.calls += 1
        return Doc(value=sorted(staged), note=";".join(feedback))


class Picky:
    """Improves by one each attempt; fails validation until its third attempt."""

    name, version = "picky", "1"

    def __init__(self):
        self.n = 0

    async def run(self, staged, *, feedback, knobs):
        self.n += 1
        return Doc(value=self.n - 1, note=" | ".join(feedback))


class Leaky:
    """A Test writer that somehow emits the oracle's canary (e.g. memorized from a shared cache)."""

    name, version = "leaky", "1"
    canary = ""

    async def run(self, staged, *, feedback, knobs):
        return Doc(value="checks", note=Leaky.canary)


class Flaky:
    name, version = "flaky", "1"

    def __init__(self):
        self.n = 0

    async def run(self, staged, *, feedback, knobs):
        self.n += 1
        if self.n < 3:
            raise RoleInfraError("provider 503")
        return Doc(value="ok")


def orch(tmp_path, roles, cap=1.0, observe=lambda: 0.0):
    return Orchestrator(Store(tmp_path / "s"), Journal(tmp_path / "j.jsonl"), Ledger(cap, observe=observe), roles)


async def test_dag_order_staging_and_resume(tmp_path):
    echo = Echo()
    o = orch(tmp_path, {"echo": echo})
    plan = Plan("t1", "g", [Node("b", "echo", {"x": "@a", "secret": "@a"}, visible=["x"]), Node("a", "echo", {"seed": "=1"})])
    assert [n.id for n in plan.order()] == ["a", "b"]
    r = await o.run_plan(plan)
    assert r.state == "constructed" and r.outputs["b"]["value"] == ["x"]  # 'secret' was an input but not visible: never staged
    o2 = orch(tmp_path, {"echo": echo})  # a new orchestrator on the same journal resumes: nothing re-runs
    calls = echo.calls
    r2 = await o2.run_plan(plan)
    assert echo.calls == calls and all(n.reused for n in r2.nodes.values())
    with pytest.raises(ValueError):
        Plan("t", "g", [Node("a", "echo", {"x": "@b"}), Node("b", "echo", {"y": "@a"})]).order()


async def test_retries_carry_feedback_and_fail_closed(tmp_path):
    def need_two(out, staged):
        return [] if out.value >= 2 else [f"attempt {out.value}: missing check for property p{out.value}"]

    o = orch(tmp_path, {"picky": Picky()})
    r = await o.run_plan(Plan("t", "g", [Node("x", "picky", {}, validators=[need_two])]))
    assert r.state == "constructed" and r.nodes["x"].attempts == 3 and "property p1" in r.outputs["x"]["note"]
    o2 = orch(tmp_path / "b", {"picky": Picky()})
    r2 = await o2.run_plan(Plan("t", "g", [Node("x", "picky", {}, validators=[need_two], max_attempts=2), Node("y", "picky", {"x": "@x"})]))
    assert r2.state == "rejected" and r2.nodes["x"].status == "failed" and r2.nodes["y"].status == "skipped"


async def test_infra_errors_are_not_content_attempts(tmp_path):
    o = orch(tmp_path, {"flaky": Flaky()})
    r = await o.run_plan(Plan("t", "g", [Node("x", "flaky", {}, max_attempts=1)]))
    assert r.nodes["x"].status == "done" and r.nodes["x"].attempts == 1


class Spender:
    name, version = "spender", "1"
    spent = 0.0

    async def run(self, staged, *, feedback, knobs):
        Spender.spent += 0.04  # what the role's LLM clients report
        return Doc(value="x")


async def test_budget_reservation_and_overrun(tmp_path):
    Spender.spent = 0.0
    o = orch(tmp_path, {"echo": Spender()}, cap=0.05, observe=lambda: Spender.spent)
    r = await o.run_plan(Plan("t", "g", [Node("a", "echo", {}, estimate_usd=0.04), Node("b", "echo", {"x": "@a"}, estimate_usd=0.04)]))
    assert r.nodes["a"].status == "done" and r.nodes["b"].status == "abandoned" and r.state == "abandoned"
    led = Ledger(0.01)
    led.reserve("n", 0.005)
    with pytest.raises(BudgetExhausted):
        led.settle("n", 0.02)


async def test_canary_breach_through_hidden_inputs(tmp_path):
    echo = Echo()
    o = orch(tmp_path, {"echo": echo, "leaky": Leaky()})
    plan = Plan("t", "g", [Node("oracle", "echo", {"seed": "=1"}, canary_out=True),
                           Node("checks", "leaky", {"spec": "=2"}),  # the Test writer: no path to the oracle
                           Node("adversary", "leaky", {"oracle": "@oracle"})])  # the Adversary may see it
    first = await o.run_plan(Plan("t0", "g", plan.nodes[:1]))
    Leaky.canary = o.canaries["oracle"]
    r = await o.run_plan(Plan("t", "g", plan.nodes))
    assert r.nodes["checks"].status == "breach" and "oracle" in r.nodes["checks"].findings[0]
    assert r.nodes["adversary"].status == "done" and first.state == "constructed"


def test_learnability_and_bands():
    from crucible.forge.learnability import BANDS, advise, estimate, learnability

    assert learnability(0.5) == 1.0 and learnability(0.0) == 0.0 and learnability(1.0) == 0.0
    r = estimate([True, False, False, False, True, False, False, False], model="m", harness="h")
    assert r.p == 0.25 and r.lo < 0.25 < r.hi
    assert advise(estimate([True] * 16, model="m", harness="h"), BANDS["train"])[0].direction == "harder"
    assert advise(estimate([False] * 16, model="m", harness="h"), BANDS["train"])[0].direction == "easier"
    assert advise(estimate([True, False] * 8, model="m", harness="h"), BANDS["train"]) == []
    early = advise(estimate([False] * 4, model="m", harness="h"), BANDS["train"], final=False)
    assert early[0].direction == "sample_more"  # 0/4: the interval [0, 0.49] still overlaps the band
    bench = advise(estimate([True, False, False, False] * 4, model="frontier", harness="h"), BANDS["bench"])
    assert bench and bench[0].direction == "harder"  # pass@1 0.25 is above the benchmark ceiling 0.2


def test_dedup_layers():
    from crucible.forge.dedup import Candidate, dedup

    a = Candidate("a", instruction="Refund the duplicate charge for order O-7101 on ticket 21.", footprint=["refunds:21:O-7101", "notes:21"],
                  checks=["sql:refunds", "sql:notes", "sql:status"], content="x1")
    b = Candidate("b", instruction="Refund the duplicate charge for order O-7101 on ticket 21.  ", footprint=["refunds:21:O-7101", "notes:21"],
                  checks=["sql:refunds", "sql:notes"], content="x2")
    c = Candidate("c", instruction="Escalate ticket 23 for a carrier trace.", footprint=["tickets:23"], checks=["sql:status"], content="x3")
    d = Candidate("d", instruction="other words", footprint=["x"], checks=["sql:a", "sql:b", "sql:c", "sql:d", "sql:e"], content="x1")
    out = dedup([a, b, c, d])
    assert out["kept"] == ["a", "c"] and out["dropped"]["b"]["layer"] == "L1_fingerprint" and out["dropped"]["d"]["layer"] == "L0_content"
