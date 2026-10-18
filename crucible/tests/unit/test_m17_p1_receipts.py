"""P1 receipts (factory/receipts.py; docs/research/swarm-pipeline/00 §13, 01 §4.2): the seeded-defect run catches every
planted task defect at its designated state and every machine-layer defect as designed, a clean control passes every gate,
and the receipt checker is mutation-tested (Prime's twin-test discipline): wrong outcomes, edits and a missing
``not_verified`` are rejected."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from crucible.factory.machine import MachineRun, StateContext
from crucible.factory.receipts import (
    EXPECTED_SEEDED,
    TASK_DEFECTS,
    check_receipt,
    make_receipt,
    run_receipt,
    run_seeded_defects,
    sha256_json,
    validate,
)


@pytest.fixture(scope="module")
def receipt(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    import asyncio

    root = tmp_path_factory.mktemp("seeded")
    return asyncio.run(run_seeded_defects(root, receipt_path=root / "receipt.json"))


def test_seeded_defect_run_passes_for_the_machine_layer(receipt: dict[str, Any]) -> None:
    validate(receipt, EXPECTED_SEEDED)
    cases = {c["case"]: c for c in receipt["cases"]}
    assert set(TASK_DEFECTS) | {"clean_control"} <= set(cases) and all(c["ok"] for c in cases.values())
    for defect, state in TASK_DEFECTS.items():
        assert cases[defect]["observed"] == {"status": "rejected", "caught_at": state}
        assert cases[defect]["evidence"]["ok"] is False  # the catching state's verdict, with its evidence
    assert cases["clean_control"]["observed"]["status"] == "done"
    assert cases["flaky_test"]["evidence"]["evidence"]["upper"] > 0.05 >= cases["clean_control"]["evidence"]["s10_flake"]["evidence"]["upper"]
    assert receipt["not_verified"] and receipt["versions"]["python"] and receipt["verified"] is True


def mutate(receipt: dict[str, Any], fn: Any, *, rehash: bool = True) -> dict[str, Any]:
    r = copy.deepcopy(receipt)
    fn(r)
    if rehash:
        r["sha256"] = sha256_json({k: v for k, v in r.items() if k != "sha256"})
    return r


def case(r: dict[str, Any], name: str) -> dict[str, Any]:
    return next(c for c in r["cases"] if c["case"] == name)


@pytest.mark.parametrize("name,fn", [
    ("caught at the wrong state", lambda r: case(r, "hinting_prompt")["observed"].update(caught_at="s5_verifier")),
    ("a defect reached done", lambda r: case(r, "leaky_world")["observed"].update(status="done", caught_at=None)),
    ("a designation moved to match", lambda r: [case(r, "flaky_test")["observed"].update(caught_at="s12_hack"),
                                                case(r, "flaky_test")["designated"].update(caught_at="s12_hack")]),
    ("the clean control was dropped", lambda r: r.update(cases=[c for c in r["cases"] if c["case"] != "clean_control"])),
    ("evidence edited", lambda r: case(r, "mislabelled_cause")["evidence"].update(ok=True)),
    ("a counted dead worker", lambda r: case(r, "dead_worker")["observed"].update(attempts=2, task_fails=1)),
    ("an exhausted loop ended done", lambda r: case(r, "unbounded_loop")["observed"].update(status="done", decision=None)),
    ("not_verified emptied", lambda r: r.update(not_verified=[])),
    ("verified flag forced", lambda r: [case(r, "missing_capture").update(ok=False), r.update(verified=True)]),
])
def test_checker_rejects_wrong_outcomes(receipt: dict[str, Any], name: str, fn: Any) -> None:
    bad = mutate(receipt, fn)
    assert check_receipt(bad, EXPECTED_SEEDED), name
    with pytest.raises(AssertionError):
        validate(bad, EXPECTED_SEEDED)


def test_written_receipt_round_trips_and_an_edit_without_rehash_is_caught(tmp_path: Path, receipt: dict[str, Any]) -> None:
    from crucible.factory.receipts import write_receipt

    on_disk = json.loads(write_receipt(tmp_path / "r.json", receipt).read_text())
    assert check_receipt(on_disk, EXPECTED_SEEDED) == []
    bad = mutate(on_disk, lambda r: r["versions"].update(python="2.7"), rehash=False)
    assert any("edited after writing" in p for p in check_receipt(bad))


async def test_every_run_writes_a_receipt(tmp_path: Path) -> None:
    async def a(ctx: StateContext) -> dict:
        return {"out": "x"}

    m = {"states": [{"id": "a", "entry": True, "fn": "a", "outputs": [{"name": "out", "type": "text"}]}, {"id": "done", "terminal": "done"}],
         "transitions": [{"from": "a", "to": "done"}]}
    run = MachineRun(m, run_dir=tmp_path / "r", handlers={"a": a}, run_id="r")
    await run.run()
    rec = run_receipt(run, scope="one state", not_verified=["nothing model-backed ran"], seeds={"seed": 7})
    assert check_receipt(rec) == [] and rec["runs"]["r"]["status"] == "done" and rec["versions"]["machine_hash"] == run.hash
    assert rec["runs"]["r"]["states"]["a"] == {"entries": 1, "attempts": 1, "settles": 1, "task_fails": 0, "infra_fails": 0, "infra_respawns": 0,
                                               "recoveries": {"reset": 0, "reconcile": 0}, "capture_failed": 0, "snapshots": 0, "tokens": 0, "usd": 0.0}
    assert check_receipt(make_receipt(scope="x", runs={}, not_verified=[" "]))  # a blank not_verified entry is no statement
