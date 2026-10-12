"""M9 audit regressions (docs/audit/2026-10-12/M09-accept.md): one test per reproduced defect (V1-V18) and per pick.

Every test fails on the M8 audit commit (2bf33e9). Symbols the audit added are imported inside the tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from crucible.accept.model import Status
from crucible.accept.registry.fixtures import FakeTask, Neg, good
from crucible.accept.run import accept_task


def _negs():
    return [Neg("n1", solution="bad1"), Neg("naive_floor", kind="floor", solution="bad2"),
            Neg("alt", kind="valid_alternative", expected="pass", expected_gate=None, solution="alt")]


def _mut():
    from crucible.accept.mutation.base import Mutant

    return [Mutant("drop_step/0", "drop_step", "mutant-0")]


async def _one(t, check, tmp_path, **kw):
    r = await accept_task(t, only=[check], cache_root=tmp_path, reg_version="r", reuse=kw.pop("reuse", False), **kw)
    return r.results[check]


# ---------------------------------------------------------------- P0


async def test_v1_agent_image_leak_fails_a5(tmp_path, monkeypatch):
    from crucible.accept.model import Need
    from crucible.verify import leak_audit
    from crucible.verify.leak_audit import LeakReport

    def fake_audit(image, **kw):
        rep = LeakReport()
        rep.findings.append({"file": "solution/solve.sh", "kind": "hidden_string", "match": "SECRET-ANSWER-1234"})
        return rep

    monkeypatch.setattr(leak_audit, "audit_image", fake_audit)
    t = FakeTask(grade_fn=good, negatives=_negs(), extra={"images": {"agent": "img"}})
    r = await accept_task(t, only=["A5"], cache_root=tmp_path, reg_version="r", needs=[Need.DOCKER], reuse=False)
    assert r.results["A5"].status == Status.FAIL


async def test_v2_user_paths_report_the_full_contract(tmp_path):
    from crucible.accept.run import verdict

    t = FakeTask(grade_fn=good, negatives=_negs(), extra={"mutants": _mut})
    rep = await accept_task(t, cache_root=tmp_path, reg_version="r")
    v = verdict(rep)
    assert v["state"] in ("provisional", "rejected") and v["blocking"]  # never "accepted" without the full contract
    src = (Path(__file__).parents[2] / "src" / "crucible" / "studio" / "build.py").read_text()
    assert "accept_task" in src and "from crucible.accept.contract import accept" not in src


# ---------------------------------------------------------------- P1


async def test_v3_zero_mutants_and_process_mutants_are_not_a_pass(tmp_path):
    t = FakeTask(grade_fn=good, negatives=_negs(), extra={"mutants": list})
    r = await _one(t, "A3", tmp_path)
    assert r.status == Status.FAIL and any("no scored mutant" in x for x in r.reasons)
    from crucible.accept.mutation.base import Mutant

    def g(sol, run):  # a dropped approval message scores 1 and leaves the end state alone (B.15)
        return (1.0, {"g": "pass"}) if sol in ("oracle", "alt", "no-message") else (0.0, {"g": "fail"})

    class SameState(FakeTask):
        async def grade(self, solution, *, run=0):
            rep, _ = await super().grade(solution, run=run)
            return rep, "fp:end" if solution in ("oracle", "no-message") else f"fp:{solution}"

    t2 = SameState(grade_fn=g, negatives=_negs(), extra={"mutants": lambda: [Mutant("drop_message/0", "drop_message", "no-message")]})
    r2 = await _one(t2, "A3", tmp_path / "2")
    assert r2.status == Status.FAIL and not r2.measured["mutation"]["equivalent"]


def test_v4_repair_verifies_the_negatives_and_attacks():
    from crucible.accept.repair import ForgeWorkspace

    ws = ForgeWorkspace({}, rebuild=lambda t, c: t)
    assert {"A2", "A3", "A13"} <= set(ws.checks)


def test_v5_key_covers_every_deciding_module():
    from crucible.accept import key

    assert {"accept", "verify", "forge/learnability.py", "world/leaks.py"} <= set(key.DECIDING)
    assert (key.PKG / "accept" / "checks" / "core.py").exists()


async def test_v6_facts_require_their_capabilities(tmp_path):
    t = FakeTask(grade_fn=good, negatives=_negs(), extra={"facts": {"reference_build": True}, "mutants": _mut})
    assert (await _one(t, "A4", tmp_path)).status == Status.FAIL
    t2 = FakeTask(grade_fn=good, negatives=_negs(), extra={"images_ref": {"verifier": "sha256:" + "a" * 64}})
    r = await accept_task(t2, only=["A9"], cache_root=tmp_path / "2", reg_version="r", reuse=False)
    assert r.results["A9"].status == Status.FAIL and "no pinned digest" in r.results["A9"].reasons[0]


async def test_v7_backfill_quarantines_errors(tmp_path):
    from crucible.accept.registry.model import load_registry
    from crucible.accept.registry.run import backfill, quarantined

    d = next(x for x in load_registry().defects if x.detector.check.split(".")[0] == "A1" and not x.detector.applies_to)

    def boom(sol, run):
        raise RuntimeError("grader crashed")

    out = await backfill(d, [FakeTask(task_id="t/err", grade_fn=boom, negatives=_negs())], cache_root=tmp_path, reg_version="r")
    assert out == {"t/err": "quarantined"} and quarantined(tmp_path) == {"t/err"}


async def test_v9_red_team_checks_need_controls(tmp_path):
    from crucible.accept.checks.extended import Attack

    t = FakeTask(grade_fn=good, negatives=_negs(), extra={"attacks": [Attack("wrong", "x", "bad-solution", "g", None)]})
    assert (await _one(t, "A13", tmp_path)).status == Status.FAIL

    async def audit():
        return {"net": "skip", "root": "skip"}

    t2 = FakeTask(grade_fn=good, negatives=_negs(), extra={"isolation_audit": audit, "tamper_attempts": []})
    r = await _one(t2, "A6", tmp_path / "2")
    assert r.status == Status.FAIL and any("passed no item" in x for x in r.reasons)


async def test_v10_na_is_derived_not_asserted(tmp_path):
    t = FakeTask(grade_fn=good, negatives=_negs(), extra={"attacks": [], "declared_env": {"kind": "world"}})
    t.attack_surface = "none"
    assert (await _one(t, "A13", tmp_path)).status == Status.FAIL
    t2 = FakeTask(grade_fn=good, negatives=_negs(), extra={"solver_trajectories": [["oracle"]]})
    r = await _one(t2, "A16", tmp_path / "2")
    assert r.status == Status.FAIL and "no property audit" in r.reasons[0]


async def test_v11_a12_runs_after_cached_a1_a3(tmp_path):
    t = FakeTask(grade_fn=good, negatives=_negs(), extra={"mutants": _mut})
    await accept_task(t, only=["A1", "A3"], cache_root=tmp_path, reg_version="r")
    r = await accept_task(t, only=["A12"], cache_root=tmp_path, reg_version="r")
    assert r.results["A12"].status == Status.NA and r.results["A12"].na_reason == "single_tier"  # timings found, not "unavailable"


# ---------------------------------------------------------------- P2


async def test_v8_v12_grade_store_by_content_and_torn_files(tmp_path):
    from crucible.accept.cache import Cache
    from crucible.accept.checks.base import _content

    f = tmp_path / "oracle.patch"
    f.write_text("a")
    h1 = json.dumps(_content(f))
    f.write_text("b")
    assert json.dumps(_content(f)) != h1
    c = Cache(tmp_path / "c")
    (c.dir("k") / "A1@3.json").write_text('{"id": "A1", "vers')
    assert c.get("k", "A1", 3) is None and not (c.dir("k") / "A1@3.json").exists()


async def test_v13_admission_edges():
    from crucible.train.admission import screen

    rep = await screen({"s": ["half"]}, lambda s, p, i: (0.5, "fail"), k=4)
    assert rep.screens[0].verdict == "broken"  # a partial reward is not a passing reference
    rep2 = await screen({"s": ["d"]}, lambda s, p, i: {"reward": 1.0}, k=4)
    assert rep2.screens[0].verdict == "stable"  # a dict verdict no longer aborts the screen
    calls = {"n": 0}

    def once_infra(s, p, i):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("node")
        return 1.0

    rep3 = await screen({"s": ["x"]}, once_infra, k=4)
    assert rep3.flake_upper_95() > rep2.flake_upper_95()  # fewer verdicts, a wider bound


async def test_v14_node_masked_fresh_grade_is_unmeasured(tmp_path):
    from crucible.accept.registry.fixtures import report

    async def fresh(sol, *, seed):
        return report(0.0, {}, infra={"status": "masked", "failure_kind": "node_contention"}), "fp"

    t = FakeTask(grade_fn=good, negatives=_negs(), extra={"fresh_gates": ["fresh"], "grade_fresh": fresh})
    assert (await _one(t, "A1", tmp_path)).status == Status.ERROR


async def test_v15_a10_fixed_n_and_the_shared_bench_band():
    from types import SimpleNamespace

    from crucible.accept.checks.base import REGISTRY
    from crucible.accept.model import Need
    from crucible.accept.policy import Policy

    class Bench:
        split = "bench"

        def calibration_runs(self):
            return [{"policy": "llm:m", "reward": 0.0, "failing": ["g"]}, {"policy": "llm:m", "reward": 0.0, "failing": ["g"]}]

    ctx = SimpleNamespace(policy=Policy(), needs={Need.PAID}, budget_usd=0.0)
    out = await REGISTRY["A10"].run(Bench(), ctx)
    assert out.status == Status.FAIL and "upper bound" in out.reasons[0]  # 0/2 is not shown hard (upper bound 0.66)

    class Record:
        split = "train"

        def calibration_runs(self):
            return [{"policy": "llm:m", "reward": 1.3, "failing": []}, {"policy": "llm:m", "reward": 0.0, "failing": ["g"]}]

    assert (await REGISTRY["A10"].run(Record(), ctx)).measured["p"] == 0.5  # a record-beating 1.3 is a success


def test_v16_overrides_cannot_reach_signoff_or_another_key():
    from crucible.accept.model import NO_OVERRIDE, Override

    assert {"S1", "A14", "A16"} <= NO_OVERRIDE
    assert "key_digest" in Override.model_fields


def test_v18_mutants_are_stratified_by_family():
    from crucible.accept.mutation import actions

    oracle = [{"kind": "tool_call", "function_name": "tickets.set_status", "arguments": {"ticket_id": i, "status": "resolved" if i % 2 else "open"}}
              for i in range(30)] + [{"kind": "message", "to": "supervisor", "text": "approve 1 USD"}]
    fams = {m.family for m in actions.generate(oracle, limit=10)}
    assert {"drop_step", "numeric_shift", "drop_message"} <= fams
