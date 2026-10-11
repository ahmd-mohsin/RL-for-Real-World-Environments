"""m13 G-6: forge economics. The forge report lists $ per accepted task per operator; a generator forced to 0% yield is
paused after 20 decided attempts (and files one issue); a family's $ cap pauses a costly generator; the diversity quota
defers a herding operator; node cost counts every attempt (retries included)."""

from __future__ import annotations

import math

from pydantic import BaseModel

from crucible.core.store import Store
from crucible.forge.budget import Ledger
from crucible.forge.economy import StopRules, YieldBook
from crucible.forge.orchestrator import Journal, Node, Orchestrator, Plan


class Doc(BaseModel):
    value: int = 0


class Spender:
    """A role that spends 0.01 USD per call (observed through the ledger, as the LLM clients report it)."""

    name, version = "spender", "1"
    spent = 0.0

    async def run(self, staged, *, feedback, knobs):
        Spender.spent += 0.01
        return Doc(value=int(knobs.get("v", 0)))


def _orch(tmp_path, book, cap=100.0):
    Spender.spent = 0.0
    return Orchestrator(Store(tmp_path / "s"), Journal(tmp_path / "j.jsonl"), Ledger(cap, observe=lambda: Spender.spent), {"spender": Spender()}, book=book)


def _never_valid(out, staged):
    return ["the checks do not kill the negatives"]


def _plan(i, gen, *, bad=False, op="", seed="", family="toy"):
    return Plan(f"{gen}/t{i}", gen, [Node("n", "spender", {"s": f"={gen}{i}"}, validators=[_never_valid] if bad else [], max_attempts=3)],
                knobs={"v": i}, operator=op or gen, seed=seed or f"seed{i}", family=family)


async def test_zero_yield_generator_is_paused_after_20_and_report_has_cost_per_accepted(tmp_path):
    book = YieldBook(path=tmp_path / "economy.jsonl")
    o = _orch(tmp_path, book)
    good = await o.run_plans([_plan(i, "good", op="chain") for i in range(6)], concurrency=1)
    for i, r in enumerate(good):
        assert r.state == "constructed"
        book.record_acceptance(r.task_id, accepted=i % 2 == 0, hacks=1 if i == 1 else 0)
    bad = await o.run_plans([_plan(i, "bad", bad=True, op="fuse") for i in range(30)], concurrency=1)
    states = [r.state for r in bad]
    assert states[:20] == ["rejected"] * 20 and states[20:] == ["paused"] * 10  # paused right after the 20th decided attempt
    assert "yield 0% < 10% after 20" in book.paused("bad") and book.paused("good") is None
    assert len(book.issues) == 1 and book.issues[0]["generator"] == "bad"
    rep = book.report()
    chain, fuse = rep["by_operator"]["chain"], rep["by_operator"]["fuse"]
    assert chain["accepted"] == 3 and abs(chain["usd_per_accepted"] - 0.06 / 3) < 1e-9 and abs(chain["hack_found_rate"] - 1 / 6) < 1e-9
    assert fuse["accepted"] == 0 and fuse["usd_per_accepted"] == math.inf and abs(fuse["usd"] - 20 * 3 * 0.01) < 1e-9  # 3 attempts each, all paid
    assert rep["stages"] == {"attempted": 26, "constructed": 6, "accepted": 3, "admitted": 0} and rep["paused"] == {"bad": book.paused("bad")}
    again = YieldBook(path=tmp_path / "economy.jsonl")  # a resumed forge keeps counts and pauses
    assert again.paused("bad") and again.report()["by_operator"]["chain"]["accepted"] == 3
    again.resume("bad")
    assert again.paused("bad") is None and "bad" not in again.report()["by_generator"]


async def test_pending_and_abandoned_do_not_count_against_yield():
    book = YieldBook(StopRules(min_attempts=3))
    for i in range(5):
        book.record_construction(f"t{i}", "g")  # constructed, M9 pending
    book.record_construction("ab", "g", state="abandoned")
    assert book.paused("g") is None and book.report()["by_generator"]["g"]["decided"] == 0
    for i in range(3):
        book.record_acceptance(f"t{i}", accepted=False)
    assert book.paused("g")


async def test_family_cost_cap_pauses_a_costly_generator():
    book = YieldBook(StopRules(usd_per_accepted_cap={"coding": 5.0}))
    book.record_construction("a", "pricey", family="coding", cost_usd=3.0)
    book.record_acceptance("a", accepted=True)
    assert book.paused("pricey") is None
    book.record_construction("b", "pricey", family="coding", cost_usd=4.0)  # 7 USD for one accepted task
    assert "exceeds the family cap" in book.paused("pricey")
    book.record_construction("c", "free", family="coding", cost_usd=6.0)  # nothing accepted, spend already above the cap
    assert book.paused("free")


async def test_diversity_quota_defers_herding_and_releases():
    book = YieldBook(StopRules(quota_floor=10, max_share=0.3))
    ops = ["chain", "fan_in", "fuse", "perturb"]
    for i in range(12):
        tid = f"t{i}"
        book.record_construction(tid, "g", operator=ops[i % 4], seed=f"s{i}", family="toy")
        book.record_acceptance(tid, accepted=True)
        ok, why = book.admit(tid)
        assert ok, why  # 3 of 12 each: 25%
    book.record_construction("h", "g", operator="chain", seed="s99", family="toy")
    book.record_acceptance("h", accepted=True)
    ok, why = book.admit("h")
    assert not ok and "operator 'chain' would hold 31%" in why[0]  # 4 of 13
    assert book.report()["deferred"] == ["h"] and book.release() == []
    for i in range(3):
        tid = f"x{i}"
        book.record_construction(tid, "g", operator="stitch", seed=f"y{i}", family="toy")
        book.record_acceptance(tid, accepted=True)
        assert book.admit(tid)[0]
    assert book.release() == ["h"]  # 4 of 16 = 25%


async def test_node_cost_counts_every_attempt(tmp_path):
    book = YieldBook()
    o = _orch(tmp_path, book)
    r = await o.run_plan(_plan(0, "g", bad=True))
    assert r.state == "rejected" and abs(r.nodes["n"].cost_usd - 0.03) < 1e-9 and abs(book.rows["g/t0"].cost_usd - 0.03) < 1e-9
