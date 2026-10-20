"""Phase A: the root curator wired to the family's task repo (``factory/root.py`` + P1's ``handoff.TaskRepo``): leases on
coverage cells with in-flight announcements checked first, free abstention, and the 40% herding guard at the S17 merge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crucible.factory.handoff import TaskRepo
from crucible.factory.machine import MachineRun
from crucible.factory.root import Cell, CoverageMatrix, RootCurator, slug
from crucible.forge.budget import Ledger

INTENT = {"id": "intent-pa"}
ACCEPTING = {"states": [{"id": "select", "entry": True, "fn": "select", "outputs": [{"name": "card", "type": "json"}]},
                        {"id": "gate", "fn": "gate", "outputs": [{"name": "verdict", "type": "json"}]},
                        {"id": "accepted", "terminal": "done"}],
             "transitions": [{"from": "select", "to": "gate"}, {"from": "gate", "to": "accepted"}]}
ABSTAINING = {"states": [{"id": "lease", "entry": True, "fn": "lease", "outputs": [{"name": "lease", "type": "json"}]},
                         {"id": "work", "fn": "work", "outputs": [{"name": "x", "type": "json"}]},
                         {"id": "done", "terminal": "done"}, {"id": "abstained", "terminal": "rejected"}],
              "transitions": [{"from": "lease", "to": "work", "when": "lease.ok == true"}, {"from": "lease", "to": "abstained", "when": "lease.ok == false"},
                              {"from": "work", "to": "done"}]}


def cells(parents: list[str]) -> CoverageMatrix:
    return CoverageMatrix([Cell("change", f"s{i:02d}", dict(INTENT), surface_spec={"parent": p}) for i, p in enumerate(parents)])


def accepting_run(cell: Cell, run_id: str, run_dir: Path) -> MachineRun:
    async def select(ctx: Any) -> dict:
        return {"card": {"id": f"task-{cell.surface}", "status": "accepted", "lineage": {"template": cell.surface_spec["parent"]}}}

    async def gate(ctx: Any) -> dict:
        return {"verdict": {"ok": True}}

    return MachineRun(ACCEPTING, run_dir=run_dir, handlers={"select": select, "gate": gate}, run_id=run_id)


def root_of(tmp_path: Path, m: CoverageMatrix, make_run: Any = accepting_run, **kw: Any) -> RootCurator:
    # one generator per cell: the yield book's operator quota stays out of the way (the herding guard is the rule here)
    return RootCurator(tmp_path, m, make_run=make_run, ledger=Ledger(5.0), concurrency=1, max_surface_share=1.0, max_attempts_per_cell=2,
                       generator=lambda c: f"gen-{c.surface}", **kw)


def repo_of(root: RootCurator) -> TaskRepo:
    return root.task_repo(INTENT["id"])  # type: ignore[return-value]


def retired(repo: TaskRepo) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((repo.root / "retired").glob("*.json"))]


async def test_leases_announce_the_cell_and_merges_retire_it_accepted_with_its_parent(tmp_path: Path) -> None:
    root = root_of(tmp_path, cells(["tpl-a", "tpl-b"]))
    rep = await root.run()
    repo = repo_of(root)
    assert repo.root == (tmp_path / "family" / slug(INTENT["id"])).resolve()
    assert len(rep.merged) == 2 and repo.inflight() == []
    assert sorted((a["task_id"], a["parent"]) for a in repo.accepted()) == [("task-s00", "tpl-a"), ("task-s01", "tpl-b")]
    assert {r["outcome"] for r in retired(repo)} == {"accepted"}
    events = [json.loads(x)["event"] for x in (tmp_path / "root.jsonl").read_text().splitlines()]
    assert events.index("announce") < events.index("retire")


async def test_a_live_announcement_of_another_holder_skips_the_cell_before_any_lease(tmp_path: Path) -> None:
    m = cells(["tpl-a", "tpl-b", "tpl-c"])
    other = TaskRepo(tmp_path / "family" / slug(INTENT["id"]), INTENT["id"])
    ann = other.announce("worker-7", {"cell": m.backlog()[1].key}, ["the cell is filled"])
    root = root_of(tmp_path, m)
    rep = await root.run()
    busy = next(c for k, c in m.cells.items() if k.endswith("s01"))
    assert busy.attempts == 0 and root.leases.runs(busy.key) == []  # no lease, no attempt
    assert rep.inflight_skipped == [{"cell": busy.key, "announcements": [ann["id"]], "holders": ["worker-7"]}]
    assert len(rep.merged) == 2 and "announced by another holder" in rep.stop_reason
    # this holder's own announcement (a crashed process of the same root) is its work: resumed, not skipped
    m2 = cells(["tpl-a"])
    own = TaskRepo(tmp_path / "r2" / "family" / slug(INTENT["id"]), INTENT["id"])
    mine = own.announce("root", {"cell": m2.backlog()[0].key}, ["the cell is filled"])
    root2 = root_of(tmp_path / "r2", m2)
    rep2 = await root2.run()
    assert len(rep2.merged) == 1 and [r["id"] for r in retired(own)] == [mine["id"]] and retired(own)[0]["outcome"] == "accepted"


async def test_abstention_is_free(tmp_path: Path) -> None:
    def abstaining_run(cell: Cell, run_id: str, run_dir: Path) -> MachineRun:
        async def lease(ctx: Any) -> dict:
            return {"lease": {"ok": False, "reason": "nothing new to add"}}

        async def work(ctx: Any) -> dict:
            return {"x": {}}

        return MachineRun(ABSTAINING, run_dir=run_dir, handlers={"lease": lease, "work": work}, run_id=run_id)

    m = cells(["tpl-a"])
    root = root_of(tmp_path, m, make_run=abstaining_run)
    rep = await root.run()
    cell = next(iter(m.cells.values()))
    assert rep.abstained and rep.rejected == [] and cell.status == "abstained" and cell.attempts == 0
    assert root.book.report()["total"].get("rejected", 0) == 0 and root.book.report()["total"].get("abandoned", 0) == 0
    assert [r["outcome"] for r in retired(repo_of(root))] == ["abstained"]
    # the root itself may abstain on a cell: recorded, free, the cell leaves the backlog
    m2 = cells(["tpl-a", "tpl-b"])
    root2 = root_of(tmp_path / "r2", m2)
    first = m2.backlog()[0].key
    root2.abstain(first, "the family already covers it")
    rep2 = await root2.run()
    assert first in rep2.abstained and len(rep2.merged) == 1 and m2.cells[first].attempts == 0
    assert [r["outcome"] for r in retired(repo_of(root2))].count("abstained") == 1


async def test_the_herding_guard_holds_a_parent_over_40_percent_out_of_the_merge(tmp_path: Path) -> None:
    root = root_of(tmp_path, cells(["tpl-a"] * 5 + ["tpl-b", "tpl-c"]))
    rep = await root.run()
    repo = repo_of(root)
    parents = [a["parent"] for a in repo.accepted()]
    assert parents.count("tpl-a") == 4 and len(parents) == 6  # the fifth tpl-a task would hold 5/5 (and 5/7 at the end): never merged
    assert len(rep.herded) == 1 and len(rep.merged) == 6
    rej = [r for r in retired(repo) if r["outcome"] == "rejected"]
    assert len(rej) == 1 and "herding guard" in rej[0]["reason"] and "tpl-a" in rej[0]["reason"]


async def test_a_herded_task_is_merged_at_the_end_when_other_parents_made_room(tmp_path: Path) -> None:
    root = root_of(tmp_path, cells(["tpl-a"] * 5 + [f"tpl-{c}" for c in "bcdefghi"]))
    rep = await root.run()
    parents = [a["parent"] for a in repo_of(root).accepted()]
    assert rep.herded == [] and len(rep.merged) == 13 and parents.count("tpl-a") == 5
    assert parents.count("tpl-a") / len(parents) <= 0.40  # 5/13
    assert all(c.status == "filled" for c in root.matrix.cells.values())


async def test_a_task_the_machine_accepted_itself_is_not_counted_twice(tmp_path: Path) -> None:
    root = root_of(tmp_path, cells(["tpl-a"]))
    repo = repo_of(root)
    ann = repo.announce("machine", {"repo": "x"}, ["done"])
    repo.retire(ann["id"], "accepted", task_id="task-s00", parent="tpl-a")  # the machine's own S17 on the same repo
    rep = await root.run()
    assert len(rep.merged) == 1 and [a["task_id"] for a in repo.accepted()] == ["task-s00"]


async def test_no_task_repo_keeps_the_p1_behaviour(tmp_path: Path) -> None:
    root = root_of(tmp_path, cells(["tpl-a"] * 6), task_repo=None)
    rep = await root.run()
    assert len(rep.merged) == 6 and rep.herded == [] and not (tmp_path / "family").exists()


async def test_a_relaunched_cell_reuses_its_announcement_and_retires_it_once(tmp_path: Path) -> None:
    root = RootCurator(tmp_path, cells(["tpl-a"] * 12), make_run=accepting_run, ledger=Ledger(5.0), concurrency=1, max_surface_share=1.0,
                       max_attempts_per_cell=2, herding_cap=1.0)  # one operator: the yield book defers, and deferred cells launch again
    rep = await root.run()
    assert rep.deferred and len({r["id"] for r in retired(repo_of(root))}) == len(retired(repo_of(root)))
    assert repo_of(root).inflight() == []
