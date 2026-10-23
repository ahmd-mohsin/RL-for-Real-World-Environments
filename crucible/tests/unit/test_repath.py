"""factory.repath: a path-rule change re-applied to stored episodes, no model call and no test run."""

from __future__ import annotations

import json
from pathlib import Path

from crucible.factory.repath import repath, scan


def _attempt(root: Path, arm: str, task: str, cmd: str, *, phi: float, end_full: bool, reward: float, patch: str = "") -> Path:
    a = root / "arms" / arm / task / "rollouts" / "run" / "samples" / "s1" / "a1"
    (a / "verifier").mkdir(parents=True)
    (a / "artifacts").mkdir()
    (a / "artifacts" / "model.patch").write_text(patch)
    trace = {"events": [{"type": "tool_call", "actor": "agent", "data": {"function": "repo.shell", "arguments": {"command": cmd}}}]}
    (a / "trace.json").write_text(json.dumps(trace))
    rep = {"reward": reward, "infra": {"status": "ok"}, "gates": {"path:install": "fail", "path:network": "pass", "path:history": "pass"},
           "meta": {"phi": phi, "full_pass": False, "end_state_full_pass": end_full, "path": {"checks": [
               {"id": "path:install", "status": "fail", "severity": "gate", "evidence": [f"package manager go mod: {cmd}"]}]}}}
    (a / "verifier" / "report.json").write_text(json.dumps(rep))
    return a


def _store(root: Path, task: str) -> Path:
    t = root / "train" / "tasks" / task
    (t / "base" / "pkg").mkdir(parents=True)
    (t / "base" / "go.mod").write_text("module example.com/pkg\n")
    (t / "window.json").write_text(json.dumps({"repo": "pkg", "lang": "go", "oracle_files": {"pkg/a.go": "package pkg\n"}, "tests_b": {}}))
    return root


def test_a_scratch_module_install_is_re_pathed_to_a_full_pass(tmp_path: Path) -> None:
    store = _store(tmp_path / "store", "db-go-x")
    a = _attempt(tmp_path / "runs", "frontier", "db-go-x", "cd /tmp/r1 && go mod tidy; go run .", phi=1.0, end_full=True, reward=0.5)
    w = json.loads((store / "train" / "tasks" / "db-go-x" / "window.json").read_text())
    r = repath(a, w, store / "train" / "tasks" / "db-go-x" / "base")
    assert r is not None and r.flipped and r.changed == ["path:install"]
    assert (r.old["reward"], r.new["reward"], r.new["full_pass"]) == (0.5, 1.0, True)


def test_scan_keeps_a_real_install_failing_and_reports_only_flips(tmp_path: Path) -> None:
    store = _store(tmp_path / "store", "db-go-x")
    _attempt(tmp_path / "runs", "frontier", "db-go-x", "cd /tmp/r1 && go mod tidy", phi=1.0, end_full=True, reward=0.5)
    _attempt(tmp_path / "runs", "student", "db-go-x", "go mod tidy", phi=1.0, end_full=True, reward=0.5)  # in the repo: still an install
    res = scan([tmp_path / "runs"], store)
    assert res["attempts"] == 2 and res["flipped"] == 1 and res["full_pass_gained"] == 1 and res["reward_down"] == 0
    assert res["flips"][0]["arm"] == "frontier"
    assert res["stale_verdicts"] == 2 and res["rules_version"]  # graded before the rules version stamp: visible as stale
