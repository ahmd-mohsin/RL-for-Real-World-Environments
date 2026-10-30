"""The student-probe outcome classifier (synth.outcome) on known-label cases: scripted attempts on the fixture tasks and
the planted defects. No model is called; the panel is scripted."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from crucible.synth.falsify import FalsifyContext
from crucible.synth.falsify.agents import Panel, PanelMember
from crucible.synth.learn import Rollout
from crucible.synth.outcome import TaskEvidence, classify, confusion, export_outcomes, summary, task_evidence
from crucible.synth.perfect import export_schemas
from tests.unit import synth_fixtures as fx
from tests.unit.synth_fakes import FakeClient
from tests.unit.synth_outcome_cases import CASES, rollout_for


def run(c: Any) -> Any:
    return asyncio.run(c)


async def _label(mk: Any, plan: Any, **kw: Any) -> Any:
    t = mk()
    ro = await rollout_for(t, plan)
    ctx = FalsifyContext()
    tev = await task_evidence(t, ctx, mut=True)
    return await classify(t, ro, ctx, tev, student="scripted", **kw)


@pytest.mark.parametrize(("name", "mk", "plan", "want"), CASES, ids=[c[0] for c in CASES])
def test_known_label_cases(name: str, mk: Any, plan: Any, want: str) -> None:
    r = run(_label(mk, plan))
    assert r.label == want, (name, r.cause, r.evidence)


def test_confusion_never_calls_a_skill_gap_a_defect() -> None:
    async def go() -> list[tuple[str, str]]:
        return [(want, (await _label(mk, plan)).label) for _, mk, plan, want in CASES]

    c = confusion(run(go()))
    assert c["skill_gap_called_defect"] == 0 and c["defect_called_intended"] == 0 and c["accuracy_decided"] == 1.0


def _panel(valid: bool | None) -> Panel:
    def script(_m: Any, _n: int) -> dict[str, Any]:
        return {"valid": bool(valid), "confidence": 0.95 if valid is not None else 0.2, "why": "scripted"}

    return Panel(members=[PanelMember("a", FakeClient("nemotron", script)), PanelMember("b", FakeClient("openai", script))], lam=0.7)


def test_the_panel_decides_only_the_residue() -> None:
    t = fx.planted("D3")
    plan = ([("ap.find_invoice", {"number": "INV-1001"}), ("ap.get_receipts", {"po_id": "PO-1"}), ("ap.approve_invoice", {"invoice_id": 101})], 1000.0)

    async def go(panel: Panel | None) -> Any:
        ro = await rollout_for(t, plan)
        ctx = FalsifyContext()
        tev = await task_evidence(t, ctx)
        tev = tev.model_copy(update={"defects": {}})  # no executed defect evidence: the answer mismatch is the residue
        return await classify(t, ro, ctx, tev, panel=panel)

    assert run(go(None)).label == "uncertain"
    judged = run(go(_panel(False)))
    assert judged.label == "unintended_pass" and judged.decided_by == "panel" and judged.evidence["panel"]["decision"] == "invalid"
    assert run(go(_panel(True))).label == "intended_pass"
    abst = run(go(_panel(None)))
    assert abst.label == "uncertain" and "abstained" in abst.cause


def test_routed_masked_and_exports() -> None:
    t = fx.t1_hold()
    ctx = FalsifyContext()
    tev = run(task_evidence(t, ctx))
    proto = Rollout(i=0, passed=False, reasons=["PROTOCOL"], stop="protocol", program={"steps": []})
    r = run(classify(t, proto, ctx, tev))
    assert r.label == "uncertain" and r.cause.startswith("routed:harness")
    masked = run(classify(t, Rollout(i=1, passed=None, stop="masked"), ctx, tev))
    assert masked.label == "uncertain" and masked.decided_by == "none"
    s = summary([r, masked])
    assert s["by_label"] == {"uncertain": 2} and not s["tasks_with_unintended"]
    ex = export_outcomes([r])
    assert ex[0]["label"] == "uncertain" and "reward" not in ex[0]
    sch = export_schemas()
    assert {"OutcomeRecord", "TaskEvidence"} <= set(sch)
    assert isinstance(tev, TaskEvidence) and tev.oracle_passes
