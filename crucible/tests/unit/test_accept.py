"""M9 acceptance: the contract runner, statuses, evidence-bound cache and key binding (G6), the defect registry's power
tests (every admitted per-task detector fires on its re-injected defect and passes the clean fixture) and backfill (G5),
and the full contract on a forged toy task (G1, CPU part)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from crucible.accept.model import CheckResult, Need, Status
from crucible.accept.registry.fixtures import FakeTask, Neg, good
from crucible.accept.registry.model import Defect, load_registry
from crucible.accept.registry.run import backfill, power_tests
from crucible.accept.run import accept_task


async def test_every_admitted_detector_is_proven():
    reg = load_registry()
    assert all(d.admitted() for d in reg.defects)
    rows = await power_tests(reg)
    per_task = [r for r in rows if r["status"] not in ("per_kit",)]
    assert per_task and all(r["status"] == "proven" for r in per_task), [r for r in per_task if r["status"] != "proven"]


def negs():
    return [Neg("n1", solution="bad1"), Neg("naive_floor", kind="floor", solution="bad2"),
            Neg("alt", kind="valid_alternative", expected="pass", expected_gate=None, solution="alt")]


def counting(fn):
    calls = []

    def g(sol, run):
        calls.append((sol, run))
        return fn(sol, run)

    g.__qualname__ = "counting_good"
    return g, calls


async def test_binding_reuses_grades_and_regrades_only_the_changed_negative(tmp_path):
    g, calls = counting(good)
    t = FakeTask(grade_fn=g, negatives=negs(), extra={"mutants": list})
    r1 = await accept_task(t, only=["A1", "A2", "A3", "A4"], cache_root=tmp_path, reg_version="r")
    assert all(r1.results[c].status == Status.PASS for c in ("A1", "A2", "A3", "A4")), r1.summary()
    n1 = len(calls)
    t2 = FakeTask(grade_fn=g, negatives=[replace(negs()[0], solution="bad1-edited"), *negs()[1:]], extra={"mutants": list})
    r2 = await accept_task(t2, only=["A1", "A2", "A3", "A4"], cache_root=tmp_path, reg_version="r")
    assert not r1.current_for(r2.key) and r1.key_digest != r2.key_digest  # an edited negative makes the old acceptance stale
    assert r2.stats["new_grades"] == 1 and len(calls) == n1 + 1  # only the edited negative was graded again
    r3 = await accept_task(t2, only=["A1", "A2", "A3", "A4"], cache_root=tmp_path, reg_version="r")
    assert r3.stats["new_grades"] == 0 and r3.key_digest == r2.key_digest  # same key: every check result reused


async def test_registry_version_is_part_of_the_key(tmp_path):
    t = FakeTask(grade_fn=good, negatives=negs(), extra={"mutants": list})
    a = await accept_task(t, only=["A2"], cache_root=tmp_path, reg_version="reg:1")
    b = await accept_task(t, only=["A2"], cache_root=tmp_path, reg_version="reg:2")
    assert a.key_digest != b.key_digest  # a new defect makes every acceptance non-current until the backfill ran


async def test_statuses_missing_capability_pending_and_na(tmp_path):
    t = FakeTask(grade_fn=good, negatives=negs(), extra={"mutants": list})
    rep = await accept_task(t, cache_root=tmp_path, reg_version="r")
    st = {k: str(v.status) for k, v in rep.results.items()}
    assert st["A6"] == "fail" and "isolation_audit" in rep.results["A6"].reasons[0]  # absence never implies N/A
    assert st["A10"] == "pending" or st["A10"] == "fail"
    assert st["S1"] == "not_applicable" and rep.results["S1"].na_reason == "horizon<H4"
    assert st["A12"] == "not_applicable"  # single tier
    assert not rep.accepted and "A6" in rep.blocking()
    with pytest.raises(ValueError):
        CheckResult(id="A4", version=1, status=Status.NA, na_reason="author says so")


async def test_dependency_failure_blocks_dependents(tmp_path):
    t = FakeTask(grade_fn=lambda s, r: (0.0, {"g": "fail"}), negatives=negs(), extra={"mutants": list})
    rep = await accept_task(t, only=["A1", "A4"], cache_root=tmp_path, reg_version="r")
    assert rep.results["A1"].status == Status.FAIL and rep.results["A4"].status == Status.PENDING  # no flake measurement of a failing oracle


async def test_signoff_bound_to_key(tmp_path):
    t = FakeTask(task_id="fake/h4@1", grade_fn=good, horizon="H4")
    rep = await accept_task(t, only=["S1"], cache_root=tmp_path / "c", signoff_root=tmp_path / "s", reg_version="r")
    assert rep.results["S1"].status == Status.PENDING
    d = tmp_path / "s" / "fake__h4@1"
    d.mkdir(parents=True)
    (d / "signoff.yaml").write_text(f"reviewer: r\ndate: 2026-10-09\nkey_digest: {rep.key_digest}\ndecision: accept\n")
    rep2 = await accept_task(t, only=["S1"], cache_root=tmp_path / "c", signoff_root=tmp_path / "s", reg_version="r", reuse=False)
    assert rep2.results["S1"].status == Status.PASS
    (d / "signoff.yaml").write_text("reviewer: r\ndate: 2026-10-09\nkey_digest: old\ndecision: accept\n")
    rep3 = await accept_task(t, only=["S1"], cache_root=tmp_path / "c", signoff_root=tmp_path / "s", reg_version="r", reuse=False)
    assert rep3.results["S1"].status == Status.PENDING  # a sign-off for another key does not count


async def test_tampered_evidence_is_not_reused(tmp_path):
    t = FakeTask(grade_fn=good, negatives=negs(), extra={"mutants": list})
    rep = await accept_task(t, only=["A2"], cache_root=tmp_path, reg_version="r")
    ev = tmp_path / rep.key_digest / rep.results["A2"].evidence[0].path
    ev.write_text(json.dumps({"status": "pass", "forged": True}))
    rep2 = await accept_task(t, only=["A2"], cache_root=tmp_path, reg_version="r")
    assert rep2.results["A2"].evidence[0].sha256 != rep.results["A2"].evidence[0].sha256 or ev.read_text().find("forged") < 0


async def test_backfill_quarantines_only_failing_tasks(tmp_path):
    from crucible.accept.registry.model import Detector, Incident

    new = Defect(id="X.headroom-all-bars", incident=Incident(date="2026-10-09", found_by="test", summary="every perf bar needs 15% headroom"),
                 detector=Detector(check="A1.perf_headroom", applies_to={"has": ["perf_bars"]}))
    ok = FakeTask(task_id="t/ok", grade_fn=good, extra={"perf_bars": {"k": 0.65}, "oracle_ratios": lambda r: {"k": 0.4}, "facts": {"perf_bars": True}})
    tight = FakeTask(task_id="t/tight", grade_fn=good, extra={"perf_bars": {"k": 0.65}, "oracle_ratios": lambda r: {"k": 0.571}, "facts": {"perf_bars": True}})
    world = FakeTask(task_id="t/world", grade_fn=good, extra={"facts": {"perf_bars": False}})
    out = await backfill(new, [ok, tight, world], cache_root=tmp_path, reg_version="r+X")
    assert out == {"t/ok": "kept", "t/tight": "quarantined", "t/world": "not_applicable"}


async def test_full_contract_on_a_forged_toy_task(tmp_path):
    import pytest
    from crucible_pack_toy import PACK_ROOT
    from crucible_pack_toy.forge.world_synth import forge

    from crucible.connectors.trace_store import TraceStore
    from crucible.core.store import Store

    m3 = PACK_ROOT.parent.parent / "data" / "m3" / "store"
    if not (m3.exists() and any(i.startswith("toy/") for i in TraceStore(Store(m3)).ids())):
        pytest.skip("needs the M3 toy ticket traces (run the M3 ingest first)")

    col: list = []
    await forge(collect=col)
    t, resolve = col[1]  # t2_approval: has a supervisor approval, so the forged-approval attack applies
    rep = await accept_task(t, needs=[Need.DOCKER], cache_root=tmp_path, resolve=resolve)
    st = {k: str(v.status) for k, v in rep.results.items()}
    assert {k for k, v in st.items() if v not in ("pass", "not_applicable")} == {"A10"}, rep.summary()  # A10 needs paid runs
    a13 = rep.results["A13"].measured["attacks"]
    assert {"forged_approval", "claim_without_doing"} <= set(a13) and all(v["took_effect"] and v["reward"] == 0.0 for v in a13.values())
    a6 = rep.results["A6"].measured["tamper"]
    assert a6 and all(v["took_effect"] is False and v["reward"] == 0.0 for v in a6.values())
    assert rep.results["A3"].measured["mutation"]["killed"] > 0 and not rep.results["A3"].measured["mutation"]["survivors"]
    assert rep.results["A11"].status == Status.PASS


class _Calib:
    """A task whose frontier model passes ``rate`` of its runs; ``calibrate(k)`` resumes one journaled run (Runner)."""

    split = "train"

    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def calibration_runs(self):
        return []

    async def calibrate(self, budget_usd, *, k=2):
        self.calls.append(k)
        return [{"policy": "llm:m", "reward": float(o), "failing": [] if o else ["outcome"]} for o in self.outcomes[:k]]


async def _a10(task, *, paid=True):
    from types import SimpleNamespace

    from crucible.accept.checks.base import REGISTRY
    from crucible.accept.model import Need
    from crucible.accept.policy import Policy

    ctx = SimpleNamespace(policy=Policy(), needs=frozenset({Need.PAID} if paid else set()), budget_usd=1.0)
    return await REGISTRY["A10"].run(task, ctx)


async def test_a10_samples_more_before_rejecting_on_thin_evidence():
    # 0/2 overlaps the band ([0, 0.66]): top up 2 -> 4; 2/4 lands in the band, so the task is accepted
    t = _Calib([0, 0, 1, 1, 0, 0, 0, 0])
    out = await _a10(t)
    assert str(out.status) == "pass" and t.calls == [2, 4] and out.measured["n"] == 4
    # never passes: top up to the cap, then rejected with the reason saying so
    t = _Calib([0] * 8)
    out = await _a10(t)
    assert str(out.status) == "fail" and t.calls == [2, 4, 8] and "sampling cap (n=8)" in out.reasons[0]
    # inside the band on the first two: no extra spend
    t = _Calib([1, 0])
    assert str((await _a10(t)).status) == "pass" and t.calls == [2]
    # no budget to sample: the first runs never happen, A10 is pending (unchanged)
    assert str((await _a10(_Calib([0, 0]), paid=False)).status) == "pending"
