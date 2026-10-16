"""M17 agent Q: the root curator (factory/root.py): parallel machines over the coverage matrix, leases with no duplicate
cell work, merging only gate-accepted tasks, budget / yield stops, escalations, persistent specialists."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from crucible.factory import driver
from crucible.factory.cards import TaskCard
from crucible.factory.machine import MachineRun
from crucible.factory.root import Cell, CoverageMatrix, LeaseBook, RootCurator, Specialists
from crucible.forge.budget import Ledger
from crucible.forge.economy import StopRules, YieldBook
from crucible.forge.hacker import HackAttempt
from tests.unit.test_m17_q_driver import INTENT, OBS, FakeLLM, FakeStages, fail

ROLES = ["planner", "author", "realism", "oracle_author", "verifier_writer", "mutant_builder", "hacker", "second_verifier", "auditor"]


def matrix(n: int = 5, surfaces: tuple[str, ...] = ("ap", "ar", "gl", "tax", "payroll")) -> CoverageMatrix:
    return CoverageMatrix([Cell("refund", surfaces[i % len(surfaces)] + (str(i) if i >= len(surfaces) else ""), dict(INTENT), observations=OBS)
                           for i in range(n)])


def maker(llm: FakeLLM, stages: FakeStages, *, ledger: Ledger | None = None, **envkw: Any) -> Any:
    def make(cell: Cell, run_id: str, run_dir: Path) -> MachineRun:
        env = driver.FactoryEnv(stages=stages.stages(), tie_break=llm, **envkw)
        return driver.task_run(run_dir=run_dir, intent=cell.intent, cell=cell.as_param(), env=env, clients=dict.fromkeys(ROLES, llm),
                               observations=cell.observations, ledger=ledger, run_id=run_id)

    return make


async def test_root_runs_machines_in_parallel_and_merges_accepted_cards(tmp_path: Path) -> None:
    merged: list[dict] = []
    ledger = Ledger(cap_usd=5.0)
    root = RootCurator(tmp_path, matrix(5), make_run=maker(FakeLLM(), FakeStages(solver_delay=0.02), ledger=ledger), ledger=ledger,
                       concurrency=3, merge=merged.append, max_surface_share=1.0)
    rep = await root.run()
    assert rep.stop_reason.startswith("coverage") and len(rep.merged) == 5 and rep.rejected == []
    assert rep.max_parallel == 3  # up to the cap, never above
    cells = [lc["cell"] for lc in rep.launches]
    assert len(cells) == len(set(cells)) == 5  # each cell worked once
    for card in merged:
        assert TaskCard.model_validate(card).status == "accepted"
    assert all(v["status"] == "filled" for row in rep.matrix.values() for v in row.values())
    assert root.book.report()["total"]["accepted"] == 5


async def test_two_roots_sharing_leases_never_work_the_same_cell(tmp_path: Path) -> None:
    llm, stages = FakeLLM(), FakeStages(solver_delay=0.02)
    a = RootCurator(tmp_path / "a", matrix(6), make_run=maker(llm, stages), ledger=Ledger(5.0), concurrency=2, holder="root-a",
                    max_surface_share=1.0)
    b = RootCurator(tmp_path / "b", matrix(6), make_run=maker(llm, stages), ledger=Ledger(5.0), concurrency=2, holder="root-b",
                    max_surface_share=1.0)
    b.leases = a.leases  # one lease book (one directory) for both
    ra, rb = await asyncio.gather(a.run(), b.run())
    worked = [lc["cell"] for lc in ra.launches + rb.launches]
    assert len(worked) == len(set(worked))  # no cell had two machines
    assert len(ra.merged) + len(rb.merged) == len(set(worked))
    assert ra.merged and rb.merged  # both did work


def test_lease_book_rules(tmp_path: Path) -> None:
    t = {"now": 0.0}
    lb = LeaseBook(tmp_path, ttl_s=10, clock=lambda: t["now"])
    a = lb.acquire("c", "A")
    assert a is not None and a.run_id == "c/001" and lb.acquire("c", "B") is None
    assert lb.acquire("c", "A").run_id == "c/001"  # the same holder after a crash resumes its run
    t["now"] = 11.0
    b = lb.acquire("c", "B")
    assert b is not None and b.run_id == "c/001"  # an expired lease reopens the unfinished run
    lb.release(b)
    c = lb.acquire("c", "B")
    assert c.run_id == "c/002"
    lb.release(c, filled=True)
    assert lb.acquire("c", "A") is None and lb.filled("c")
    with pytest.raises(KeyError):
        lb.acquire("d", "A", run_id="d/009")


async def test_budget_stop_launches_nothing_more(tmp_path: Path) -> None:
    ledger = Ledger(cap_usd=0.25)
    merged: list[dict] = []
    root = RootCurator(tmp_path, matrix(6), make_run=maker(FakeLLM(usd=0.02), FakeStages(), ledger=ledger), ledger=ledger, concurrency=2,
                       merge=merged.append, max_surface_share=1.0)
    rep = await root.run()
    assert rep.stop_reason.startswith("budget") and rep.stopped
    assert len(rep.launches) < 6 and len(merged) == len(rep.merged) < 6
    assert ledger.committed <= 0.25 + 0.02 * 2 + 1e-9  # at most one in-flight call per machine past the cap


async def test_root_refuses_to_merge_a_card_the_gate_did_not_accept(tmp_path: Path) -> None:
    m = {"states": [{"id": "select", "entry": True, "fn": "select", "outputs": [{"name": "card", "type": "json"}]},
                    {"id": "accepted", "terminal": "done"}],
         "transitions": [{"from": "select", "to": "accepted"}]}

    async def select(ctx: Any) -> dict:
        return {"card": {"id": "x", "status": "accepted"}}

    merged: list[dict] = []
    root = RootCurator(tmp_path, matrix(1), make_run=lambda c, rid, d: MachineRun(m, run_dir=d, handlers={"select": select}, run_id=rid),
                       ledger=Ledger(1.0), merge=merged.append, max_attempts_per_cell=1)
    rep = await root.run()
    assert merged == [] and rep.rejected and rep.merged == []


async def test_escalations_queue_for_a_decision(tmp_path: Path) -> None:
    stages = FakeStages({"probes": [fail("verifier", "mutant passed")] * 50})
    root = RootCurator(tmp_path, matrix(2), make_run=maker(FakeLLM(), stages), ledger=Ledger(5.0), concurrency=2, max_surface_share=1.0)
    rep = await root.run()
    assert len(rep.escalated) == 2 and set(root.decisions) == set(rep.escalated) and rep.merged == []
    assert all(d["kind"] == "exhausted" for d in root.decisions.values())
    rid = rep.escalated[0]
    res = await root.answer(rid, {"action": "drop", "actor": "human:ann", "reason": "not checkable"})
    assert res.status == "rejected" and rid not in root.decisions and rid in root.out["rejected"]


async def test_marginal_yield_stops_the_root(tmp_path: Path) -> None:
    book = YieldBook(StopRules(min_attempts=2, min_yield=0.5), path=tmp_path / "yield.jsonl")
    root = RootCurator(tmp_path, matrix(5), make_run=maker(FakeLLM(), FakeStages(), keep_per_cell=0), ledger=Ledger(5.0), book=book,
                       concurrency=1, max_surface_share=1.0)
    rep = await root.run()
    assert "marginal yield" in rep.stop_reason and len(rep.launches) == 2 and rep.merged == []


async def test_surface_share_defers_the_crowded_surface(tmp_path: Path) -> None:
    m = CoverageMatrix([Cell("refund", "ap", dict(INTENT), observations=OBS, quota=6), Cell("refund", "ar", dict(INTENT), observations=OBS)])
    root = RootCurator(tmp_path, m, make_run=maker(FakeLLM(), FakeStages()), ledger=Ledger(5.0), concurrency=1, max_attempts_per_cell=10)
    rep = await root.run()
    ap = rep.matrix["refund"]["ap"]["accepted"]
    total = ap + rep.matrix["refund"]["ar"]["accepted"]
    assert rep.matrix["refund"]["ar"]["accepted"] == 1 and ap == 3 and total == 4
    assert "diversity" in rep.stop_reason  # no numeric target: the factory stops when only the crowded surface is left


async def test_specialists_persist_per_family(tmp_path: Path) -> None:
    sp = Specialists(tmp_path, factories={"world_expert": lambda w: {"world": w}})
    sp.remember_hacks("fam", "t1", ["skip_reads", "skip_reads"])
    h = sp.hacker("fam")
    assert h.memory.strategies("fam") == ["skip_reads"]
    view = {"solver_trajectories": [[{"kind": "tool_call", "function_name": "get_invoice"}, {"kind": "tool_call", "function_name": "refund"}]]}
    got = await h.propose(view)
    assert got and all(isinstance(a, HackAttempt) and a.strategy == "skip_reads" for a in got) and h.source == "memory"
    assert sp.get("world_expert", "ap") is sp.get("world_expert", "ap")
    assert Specialists(tmp_path).memory("fam").strategies("fam") == ["skip_reads"]  # persisted
