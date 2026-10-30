"""``synth.realdefects``: the real-defect set (E8) from documented failures, archives and the finance_ap finding (M5)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from crucible.synth import realdefects as rd
from crucible.synth.archive import Archive
from crucible.synth.falsify.base import Counterexample, Probe

TRACKED = Path(__file__).resolve().parents[2] / "docs" / "research" / "env-synthesis" / "runs" / "m5" / "debug_lane_defects.json"
DOCS = Path(__file__).resolve().parents[2]


def rec(**kw):
    base = {"id": "x", "task_version": "db-go-a-1", "fixed_by_version": None, "repo": "a", "split": "train", "mechanism": "changelog_hint", "detail": "d",
            "milestones_or_fixes": [], "doc": "notes/a.md", "line": 2, "quote": "announced the fix", "found_by": "checklist", "who": "U"}
    return {**base, **kw}


def test_mapping_quotes_and_heldout(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.md").write_text("# a\nthe changelog announced the fix (#12)\n")
    f = tmp_path / "r.json"
    f.write_text(json.dumps([rec(), rec(id="y", mechanism="too_easy", quote=""), rec(id="z", split="heldout", task_version="db-go-secret-1", quote="",
                                                                                     mechanism="unstated_rule"),
                             rec(id="black-ec71b918-duplicate", mechanism="other", quote="")]))
    ds = rd.load_debug_records(f, tmp_path)
    by = {d.id: d for d in ds}
    assert by["x"].quality == "P1" and by["x"].d_class == "D7" and by["x"].independent
    assert by["y"].quality == "P2" and by["y"].d_class is None
    held = by["heldout#1"]
    assert held.d_class == "D5" and held.task_version is None and "secret" not in held.model_dump_json() and held.executable.status == "heldout"
    assert by["black-ec71b918-duplicate"].quality == "P4"
    f.write_text(json.dumps([rec(quote="a quote that is not there")]))
    with pytest.raises(rd.QuoteError):
        rd.load_debug_records(f, tmp_path)


def test_tracked_records_verify_and_carry_no_heldout_content():
    ds = rd.load_debug_records(TRACKED, DOCS)  # every training quote is at its line
    raw = json.loads(TRACKED.read_text())
    assert all(r["quote"] == "" and r["task_version"] is None for r in raw if r["split"] == "heldout")
    assert sum(1 for d in ds if d.split == "heldout") == sum(1 for r in raw if r["split"] == "heldout")
    assert {d.d_class for d in ds if d.quality == "P1"} >= {"D5", "D6", "D7", "D10", "D11"}


def test_annotate_executable(tmp_path):
    store = tmp_path / "store"
    (store / "tasks" / "db-go-a-1").mkdir(parents=True)
    (store / "tasks" / "db-go-a-1" / "window.json").write_text("{}")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.md").write_text("# a\nthe changelog announced the fix (#12)\n")
    f = tmp_path / "r.json"
    f.write_text(json.dumps([rec(), rec(id="gone", task_version="db-go-b-1"), rec(id="leak", mechanism="leak_other"), rec(id="e", mechanism="too_hard")]))
    ds = rd.load_debug_records(f, tmp_path)
    rd.annotate_debug(ds, store)
    st = {d.id: d.executable.status for d in ds}
    assert st == {"x": "executable", "gone": "lost", "leak": "no_detector", "e": "no_detector"}
    assert next(d for d in ds if d.id == "x").executable.detector == "TAINT:world_leak"


def test_archive_source_and_versions(tmp_path):
    root = tmp_path / "synth"
    arch = Archive(root)
    p = Probe(cls="RED", op="pattern", site="pattern:abc", expect="property")
    arch.add_all([Counterexample(id="cx-1", domain="dom", task_id="dom/run1/t01", task_version=1, cls="RED", op="pattern", site="pattern:abc", defect="D4", probe=p),
                  Counterexample(id="cx-2", domain="dom", task_id="dom/run1/t02", task_version=1, cls="RED", op="pattern", site="pattern:abc", defect="D4", probe=p)])
    for t, v in (("t01", 1), ("t02", 3)):
        d = root / "dom" / "run1" / "tasks" / t
        d.mkdir(parents=True)
        (d / "cegis.json").write_text(json.dumps({"task": {"version": v}}))
    ds = rd.load_archive(root)
    st = {d.id: (d.executable.status, d.independent, d.found_by) for d in ds}
    assert st == {"cx-1": ("executable", False, "falsifier:RED"), "cx-2": ("lost", False, "falsifier:RED")}


def test_measure_with_injected_detector_and_summary(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.md").write_text("# a\nthe changelog announced the fix (#12)\n")
    store = tmp_path / "store"
    (store / "tasks" / "db-go-a-1").mkdir(parents=True)
    (store / "tasks" / "db-go-a-1" / "window.json").write_text("{}")
    f = tmp_path / "r.json"
    f.write_text(json.dumps([rec()]))
    rs = rd.build_set(debug_records=f, docs_root=tmp_path, debug_store=store)
    out = asyncio.run(rd.measure(rs, detect_debug=lambda d: rd.Detection(detected=True, by=["TAINT:world_leak:x"]), include=lambda d: d.source == "debug_lane"))
    s = rd.summary(out)
    assert s["detection"] == {"D7": {"n": 1, "detected": 1}} and s["by_source"] == {"debug_lane": 1, "finance_ap": 1}
    assert "| D7 | 1 |" in rd.counts_table(out)
