"""Calibration exemplars in the debug lane's builder brief (``factory.calibration``, ``generators.debug.builder_calibration``,
``scripts/lh_debug.py brief``): training tasks by level with their measured numbers and the easing knobs; held-out
tasks never enter a brief."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from crucible.factory.calibration import KNOBS, HeldOutLeak, assert_no_heldout, brief_block, exemplars, is_training, level_of, load_records, render

STUDENT = "Qwen/Qwen3.6-35B-A3B:peft:262144"


def _rec(task: str, repo: str, split: str, p: float, verdict: str, *, fair: bool = True, model: str = STUDENT) -> dict[str, Any]:
    n = 8
    return {"task": task, "repo": repo, "split": split, "model": model, "lang": "go", "faults": 3, "level": "medium", "n": n, "masked": 0,
            "passes": round(p * n), "pass_rate": p, "mean_phi": 0.5, "per_milestone": {"m01": 0.75, "m02": 0.25}, "stops": {"agent_done": 6, "max_steps": 2},
            "keep": {"verdict": verdict, "why": f"why {verdict}", "fair": fair, "frontier": {"Qwen/Qwen3.5-397B-A17B": [fair, False]}},
            "modes": {"stalled/interacting_faults": 3}}


RECS = {
    "db-go-echo-1": _rec("db-go-echo-1", "echo", "train", 0.875, "too_easy"),
    "db-python-black-2": _rec("db-python-black-2", "black", "train", 0.25, "train"),
    "db-go-go-toml-3": _rec("db-go-go-toml-3", "go-toml", "train", 0.0, "too_hard"),
    "db-go-mvdan-sh-4": _rec("db-go-mvdan-sh-4", "mvdan-sh", "train", 0.0, "unfair", fair=False),
    "db-go-kong-5": _rec("db-go-kong-5", "kong", "heldout", 0.25, "bench"),  # held out by its split, its repo and the lease
    "db-go-excelize-6": _rec("db-go-excelize-6", "excelize", "train", 0.25, "train"),  # mislabelled: the repo is held out
    "db-go-echo-7": _rec("db-go-echo-7", "echo", "train", 0.25, "world_bug"),  # a defect, not a difficulty measurement
    "db-go-echo-8": _rec("db-go-echo-8", "echo", "train", 0.25, "train", model="Qwen/Qwen3.5-397B-A17B"),  # another model
}


def test_levels_follow_the_keep_rule() -> None:
    assert [level_of(RECS[t]) for t in ("db-go-echo-1", "db-python-black-2", "db-go-go-toml-3", "db-go-mvdan-sh-4", "db-go-echo-7")] == [
        "too_easy", "in_band", "too_hard", "unfair", None]


def test_held_out_records_are_dropped_by_split_repo_or_lease() -> None:
    assert not is_training(RECS["db-go-kong-5"]) and not is_training(RECS["db-go-excelize-6"])
    assert is_training(RECS["db-go-echo-1"]) and not is_training({**RECS["db-go-echo-1"], "repo": ""})

    class Sealed:
        def check(self, rec):
            return {"ok": rec["repo"] != "echo"}

    assert not is_training(RECS["db-go-echo-1"], lease=Sealed())  # the lease alone is enough to drop it
    ex = exemplars(RECS, model="Qwen/Qwen3.6-35B-A3B")
    tasks = {e["task"] for lv in ex.values() for e in lv}
    assert tasks == {"db-go-echo-1", "db-python-black-2", "db-go-go-toml-3", "db-go-mvdan-sh-4"}
    e = ex["in_band"][0]
    assert e["pass_rate"] == 0.25 and e["budget_stops"] == 2 and e["per_milestone"] == {"m01": 0.75, "m02": 0.25} and e["mean_phi"] == 0.5


def test_the_brief_block_has_numbers_and_knobs_and_never_a_held_out_task() -> None:
    txt = brief_block("db-go-go-toml-3", "train", RECS, model=STUDENT)
    assert "THIS TASK as measured:\n- db-go-go-toml-3" in txt and "TOO EASY" in txt and "- db-go-echo-1 " in txt
    assert "pass 2/8 = 0.25, mean Φ 0.50; per milestone m01 0.75, m02 0.25; budget stops 2/8" in txt
    assert all(f"{k} {name}" in txt for k, name, _ in KNOBS) and "repro" in txt  # K4 points at the repro field
    for held in ("db-go-kong-5", "db-go-excelize-6", "kong", "excelize"):
        assert held not in txt, held
    assert brief_block("db-go-kong-5", "heldout", RECS, model=STUDENT).startswith("CALIBRATION: omitted")
    mislabelled = brief_block("db-go-excelize-6", "train", RECS, model=STUDENT)
    assert "THIS TASK" not in mislabelled and "excelize" not in mislabelled


def test_render_refuses_a_held_out_id() -> None:
    ex = exemplars(RECS, model=STUDENT)
    with pytest.raises(HeldOutLeak):
        render(ex, model=STUDENT, heldout_ids=["db-go-kong-5"], this_task={**ex["in_band"][0], "task": "db-go-kong-5"})
    with pytest.raises(HeldOutLeak):
        assert_no_heldout("see db-go-kong-5", ["db-go-kong-5"])
    assert_no_heldout("nothing sealed here", ["db-go-kong-5"])


def test_load_records_lets_later_runs_override_and_keeps_the_budget(tmp_path: Path) -> None:
    a, b = tmp_path / "debug" / "report.json", tmp_path / "debug-300" / "report.json"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_text(json.dumps({"settings": {"tool_limit": 150}, "tasks": {"db-go-echo-1": RECS["db-go-echo-1"], "db-go-go-toml-3": RECS["db-go-go-toml-3"]}}))
    b.write_text(json.dumps({"settings": {"tool_limit": 300}, "tasks": {"db-go-go-toml-3": {**RECS["db-go-go-toml-3"], "mean_phi": 0.47}}}))
    recs = load_records([a, b, tmp_path / "missing.json"])
    assert recs["db-go-go-toml-3"]["mean_phi"] == 0.47 and recs["db-go-go-toml-3"]["_tool_limit"] == 300 and recs["db-go-echo-1"]["_tool_limit"] == 150


def _lh_debug() -> Any:
    p = Path(__file__).resolve().parents[2] / "scripts" / "lh_debug.py"
    spec = importlib.util.spec_from_file_location("lh_debug_cal", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lh_debug_brief_reads_the_s15_reports(tmp_path: Path) -> None:
    mod = _lh_debug()
    rep = tmp_path / "report.json"
    rep.write_text(json.dumps({"settings": {"tool_limit": 300}, "tasks": RECS}))
    txt = mod.calibration_text("db-python-black-2", "train", [str(rep)])
    assert txt.startswith("CALIBRATION for the target student Qwen/Qwen3.6-35B-A3B") and "db-go-kong-5" not in txt
    assert mod.calibration_text("db-go-kong-5", "heldout", [str(rep)]).startswith("CALIBRATION: omitted")
