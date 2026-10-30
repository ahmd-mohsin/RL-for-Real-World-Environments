"""The perfect-task subcommands of scripts/synth.py (learnability, novelty, perfect, frontier, outcome) on a temporary
data root holding a synthesis run made of the fixture tasks. No model is called (``--dry-run``, no paraphrase/panel)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from crucible.synth.cegis import CegisReport
from tests.unit import synth_fixtures as fx

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "synth.py"


def _cli() -> Any:
    spec = importlib.util.spec_from_file_location("synth_cli_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _pool(root: Path) -> None:
    run = root / "desk-test" / "20261029-000000-fixture"
    tasks = [fx.in_domain(fx.t3_resolve_email()), fx.in_domain(fx.t6_resolve()), fx.in_domain(fx.t4_priority())]
    rows, props = [], []
    for n, t in enumerate(tasks, 1):
        d = run / "tasks" / f"t{n:02d}"
        d.mkdir(parents=True)
        d.joinpath("cegis.json").write_text(CegisReport(task_id=t.id, domain=t.domain, status="admitted", task=t).model_dump_json())
        rows.append({"n": n, "id": t.id, "status": "certified" if n < 3 else "rejected"})
        props.append({"n": n, "cell": ["a decision on one entity that needs evidence from two or more records",
                                       "several related writes across entities, chosen by a stated rule"][n % 2]})
    run.joinpath("report.json").write_text(json.dumps({"tasks": rows}))
    run.joinpath("proposals.json").write_text(json.dumps(props))


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["synth.py", *argv])
    _cli().main()


def test_subcommands_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "synth"
    _pool(root)
    store = str(tmp_path / "difficulty.jsonl")
    common = ["--data-root", str(root)]
    _main(monkeypatch, *common, "learnability", "--domains", "desk-test", "--dry-run", "--limit", "0", "--store", store, "--tag", "t")
    lines = [json.loads(x) for x in Path(store).read_text().splitlines()]
    assert len({x["key"]["task_id"] for x in lines}) == 3 and all(x["decision"] == "too_hard" for x in lines)
    run = max((root / "perfect").glob("*-perfect-t"))
    rep = json.loads((run / "learnability.json").read_text())
    assert rep["decisions"] == {"too_hard": 3} and rep["cost"]["decisions"] == 3
    _main(monkeypatch, *common, "novelty", "--domains", "desk-test", "--tag", "nov")
    nov = max((root / "perfect").glob("*-perfect-nov")) / "novelty.json"
    cal = json.loads(nov.read_text())["calibration"]
    assert cal["positives"] > 0 and 0 <= cal["theta"] <= 1
    _main(monkeypatch, *common, "perfect", "--domains", "desk-test", "--store", store, "--calibration", str(nov), "--tag", "chk")
    chk = json.loads((max((root / "perfect").glob("*-perfect-chk")) / "perfect.json").read_text())
    s = chk["by_domain"]["desk-test"]["summary"]
    assert s["candidates"] == 3 and s["valid"] == 2 and s["learnable"] == 0 and s["admit"] == 0  # the no-op student: too hard
    assert chk["by_domain"]["desk-test"]["on_intent"]["measured"] >= 1
    _main(monkeypatch, *common, "frontier", "--domains", "desk-test", "--store", store, "--tag", "fr")
    fr = json.loads((max((root / "perfect").glob("*-perfect-fr")) / "frontier.json").read_text())
    assert fr["summary"]["failed"] == 24 and fr["summary"]["by_kind"] == {"no_action": 24}
    assert fr["requests"] and fr["requests"][0]["direction"] == "ease"
    _main(monkeypatch, *common, "outcome", "--domains", "desk-test", "--store", store, "--tag", "oc")
    oc = json.loads((max((root / "perfect").glob("*-perfect-oc")) / "outcome.json").read_text())
    assert oc["summary"]["rollouts"] == 24 and set(oc["summary"]["by_label"]) <= {"intended_fail", "uncertain"}
    capsys.readouterr()
