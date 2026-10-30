"""DESIGN §7.7 item 20: the human-queue triage tool groups, ranks and exports; it decides nothing."""

from __future__ import annotations

import csv

from crucible.synth.triage import entropy, export, groups, p_invalid, read_decisions, site_pattern, triage


def _row(task: str, op: str, site: str, va: bool, ca: float, vb: bool, cb: float, passed: bool = True) -> dict:
    return {"task": task, "version": 1, "cls": "MUT" if op != "agent" else "ALT", "op": op, "site": site, "grader_passed": passed, "decision": "abstain",
            "votes": [{"family": "nemotron", "valid": va, "confidence": ca, "why": "a"}, {"family": "openai", "valid": vb, "confidence": cb, "why": "b"}],
            "instruction": f"do {task}", "attempt": {"trace": [{"tool": "x.set", "args": {"k": 1}, "result": "{\"ok\": true}"}], "answer": None}}


def test_triage_groups_ranks_and_exports_without_deciding(tmp_path):
    rows = [_row("r/t1", "PerturbValue", "s3.reason=dup", False, 0.9, True, 0.9), _row("r/t1", "PerturbValue", "s3.reason=late", False, 0.9, True, 0.9),
            _row("r/t2", "DupStep", "s2", False, 0.9, True, 0.6), _row("r/t3", "agent", "alt-0", True, 0.9, False, 0.9, passed=False)]
    rows.append(dict(rows[0]))  # one probe queued twice is one item
    report = {"tasks": [{"id": "r/t1", "status": "rejected", "reasons": ["MUT:upper=0.06>0.05", "MUT:suspects=2"]}, {"id": "r/t2", "status": "rejected", "reasons": ["TWIN:confirmed=1", "MUT:suspects=1"]}]}
    items = triage(rows, report=report)
    assert len(items) == 4 and [i.rank for i in items] == [1, 2, 3, 4]
    t1 = [i for i in items if i.task == "r/t1"]
    assert all(i.pivotal and i.similar == 2 for i in t1) and items[0].task == "r/t1" and items[0].representative
    assert next(i for i in items if i.task == "r/t2").pivotal is False  # a confirmed counterexample rejects it whatever the queue says
    assert sum(i.representative for i in items) == 3 and not items[-1].representative  # the second t1 item follows the other clusters
    assert p_invalid(rows[0]["votes"]) == 0.5 and entropy(0.5) == 1.0 and entropy(1.0) == 0.0
    assert site_pattern("PerturbValue", "s3.reason=dup") == "s3.reason=*" and site_pattern("SwapEntity", "s1.id->T-9") == "s1.id->*"
    g = groups(items)
    assert {x["reason"] for x in g} >= {"MUT/PerturbValue, grader passed, nemotron: invalid vs openai: valid"}
    paths = export(items, tmp_path / "sheet")
    with open(paths["csv"]) as fh:
        sheet = list(csv.DictReader(fh))
    assert len(sheet) == 4 and all(r["decision"] == "" for r in sheet)  # nothing decided
    assert "FAILED" in next(r for r in sheet if r["grader"] == "failed")["question"]
    md = (tmp_path / "sheet" / "review.md").read_text()
    assert md.index("Question") < md.index("probe, read after deciding")
    sheet[0]["decision"], sheet[1]["decision"], sheet[2]["decision"] = "invalid", "unsure", " Valid "
    with open(paths["csv"], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    assert read_decisions(paths["csv"]) == {sheet[0]["id"]: "invalid", sheet[2]["id"]: "valid"}
