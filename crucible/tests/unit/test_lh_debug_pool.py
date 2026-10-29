"""The debug lane's training pool (``scripts/lh_debug_pool.py``): the tracked manifest's targeting records in the
trainer's schema, the check of the pool on disk against the manifest, and relocatable clone alternates. No Docker."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3.6-35B-A3B"


def _pool():
    spec = importlib.util.spec_from_file_location("lh_debug_pool", ROOT / "scripts" / "lh_debug_pool.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _targeting(passes: int, n: int = 8) -> dict:
    return {"band": {"in_band": 0 < passes < n, "n": n, "passes": passes, "pass_rate": passes / n, "rewards": None},
            "keep": {"verdict": "train", "passes": passes, "n": n}, "source": "report.json (Tinker, 2026-10-28)"}


def _disk(tmp_path: Path, tid: str, a: str = "a1", *, findings=(), gates_a: str | None = None, split: str = "train") -> tuple[Path, dict]:
    debug = tmp_path / "debug"
    store = debug / "gen" / "store" / "train" / "tasks" / tid
    store.mkdir(parents=True)
    (store / "window.json").write_text(json.dumps({"id": tid, "a": a, "world": {"image": "crucible-lh-d:go-x-1"}}))
    for sub in ("authored", "prompts", "gates"):
        (debug / sub).mkdir(exist_ok=True)
    (debug / "authored" / f"{tid}.json").write_text(json.dumps({"findings": list(findings)}))
    (debug / "prompts" / f"{tid}.json").write_text('{"requirements": {"m01": "x"}}')
    (debug / "gates" / f"{tid}.json").write_text(json.dumps({"ok": True, "passed": ["S6", "S5", "S8", "S3", "S10"], "window_a": gates_a or a}))
    (debug / "candidates.json").write_text(json.dumps([{"id": tid, "split": split, "gates_ok": True}]))
    row = {"status": "ready", "built": {"window_a": a, "image": "crucible-lh-d:go-x-1"}, "recorded": {"window_a": a},
           "prompt_sha256": hashlib.sha256((debug / "prompts" / f"{tid}.json").read_bytes()).hexdigest(), "targeting": _targeting(4)}
    return debug, row


def test_targeting_records_are_the_trainers_schema_for_ready_tasks_only() -> None:
    from crucible.train.bridges.tinker.trainer import check_band

    p = _pool()
    m = {"student": f"{MODEL}:peft:262144", "tasks": {"db-go-a-1": {"status": "ready", "targeting": _targeting(3)},
                                                     "db-go-b-2": {"status": "blocked", "why": "recipe not recorded", "targeting": _targeting(3)}}}
    recs = p.targeting_records(m)
    assert list(recs) == ["db-go-a-1"] and recs["db-go-a-1"]["model"] == MODEL and recs["db-go-a-1"]["split"] == "train"
    assert recs["db-go-a-1"]["source"].startswith("report.json") and check_band(["db-go-a-1"], MODEL, recs) == []


def test_check_finds_a_task_that_is_not_the_recorded_one(tmp_path: Path) -> None:
    p = _pool()
    tid = "db-go-x-00000001"
    debug, row = _disk(tmp_path, tid)
    m = {"student": MODEL, "tasks": {tid: row}}
    assert p.check(m, debug) == {tid: []}
    assert any("recorded" in b for b in p.check({**m, "tasks": {tid: {**row, "recorded": {"window_a": "other"}}}}, debug)[tid])
    assert any("sha256" in b for b in p.check({**m, "tasks": {tid: {**row, "prompt_sha256": "0" * 64}}}, debug)[tid])
    assert any("out of band" in b for b in p.check({**m, "tasks": {tid: {**row, "targeting": _targeting(8)}}}, debug)[tid])
    debug2, row2 = _disk(tmp_path / "2", tid, findings=["hidden_test: x"], gates_a="old", split="heldout")
    bad = " ".join(p.check({"student": MODEL, "tasks": {tid: row2}}, debug2)[tid])
    assert "lint findings" in bad and "another start commit" in bad and "not train" in bad


def test_relocate_makes_the_clone_alternates_relative(tmp_path: Path) -> None:
    p = _pool()
    debug = tmp_path / "pool"
    src = debug / "src" / "o__x" / ".git" / "objects"
    src.mkdir(parents=True)
    info = debug / "gen" / "clones" / "x" / ".git" / "objects" / "info"
    info.mkdir(parents=True)
    (info / "alternates").write_text(f"{src}\n")
    assert p.relocate(debug) == ["gen/clones/x/.git/objects/info/alternates"]
    alt = (info / "alternates").read_text().strip()
    assert not alt.startswith("/") and (info.parent / alt).resolve() == src.resolve()
    assert p.relocate(debug) == []  # already relative: nothing to do
