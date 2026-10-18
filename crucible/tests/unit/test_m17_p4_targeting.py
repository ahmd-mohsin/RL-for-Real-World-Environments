"""P4 failure targeting (swarm-pipeline/00 §7, §4 S15; ``factory.targeting``): the statistics, the counterfactual verdicts,
the prerequisite probe, paraphrase checks and stability, band and cause mix, the S15 verdict and its machine wiring, and the
counterfactual on the toy window with scripted policies (a prerequisite-bound policy is attributed, a tool-broken one is
not)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from test_m17_w_task import make_task

from crucible.factory.longtask import ground_truth, new_file_patch
from crucible.factory.machine import MachineRun, Settle, validate_machine
from crucible.factory.targeting import (
    SIMPLIFY_KINDS,
    CounterfactualConfig,
    Outcome,
    Simplifiable,
    TargetEvidence,
    band,
    binom_cdf,
    bootstrap_ci,
    bootstrap_diff_ci,
    cause_mix,
    check_paraphrase,
    counterfactual,
    fisher_one_sided,
    fisher_power,
    outcome,
    outcomes_of,
    paraphrased_task,
    prerequisite_probe,
    probe_episode,
    seeded_tooling,
    simplify,
    split_request,
    stability,
    target_handlers,
    target_machine,
    target_verdict,
)


def oc(success: bool, phi_set: float = 0.0, *, phi: float = 0.0, reward: float = 0.0, full: bool = False, masked: bool = False) -> Outcome:
    return Outcome(success, phi_set, phi, reward, full_pass=full, masked=masked)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return make_task(tmp_path_factory.mktemp("p4t"))


# ---- statistics -----------------------------------------------------------------------------------------------------------
def test_fisher_binomial_and_power_match_hand_values() -> None:
    assert fisher_one_sided(4, 4, 0, 4) == pytest.approx(1 / 70)
    assert fisher_one_sided(3, 4, 0, 4) == pytest.approx(5 / 70)  # C(3,3)C(5,0)/C(8,4)... = P(X>=3) with 3 successes in 8
    assert fisher_one_sided(0, 4, 0, 4) == 1.0 and fisher_one_sided(2, 4, 2, 4) == pytest.approx(53 / 70)
    assert binom_cdf(0, 4, 0.8) == pytest.approx(0.2**4) and binom_cdf(1, 4, 0.8) == pytest.approx(0.2**4 + 4 * 0.8 * 0.2**3)
    assert fisher_power(0.0, 1.0, 4, 4) == pytest.approx(1.0)  # 4/4 vs 0/4 always: p = 1/70
    assert fisher_power(0.0, 0.8, 4, 4) == pytest.approx(0.8**4)  # only 4/4 is significant at n = 4
    assert fisher_power(0.0, 0.8, 8, 8) > fisher_power(0.0, 0.8, 4, 4)
    lo, hi = bootstrap_ci([0.0, 0.5, 1.0, 0.5])
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0 and bootstrap_ci([0.3] * 4) == (0.3, 0.3)
    assert bootstrap_diff_ci([0.0] * 4, [0.5] * 4) == (0.5, 0.5)


# ---- the counterfactual verdicts --------------------------------------------------------------------------------------------
def test_counterfactual_attributed_not_attributed_and_underpowered() -> None:
    a = counterfactual([oc(False)] * 4, [oc(True, 1.0)] * 4)
    assert a["verdict"] == "attributed" and a["fisher_p"] == pytest.approx(1 / 70, abs=1e-6) and a["route"] is None
    assert a["power_at_ref"] == pytest.approx(0.8**4) and a["min_simple_successes_for_significance"] == 4
    na = counterfactual([oc(False)] * 4, [oc(False)] * 3 + [oc(True, 1.0)])
    assert na["verdict"] == "not_attributed" and na["route"] == "S2" and na["simple_vs_ref_p"] < 0.05
    up = counterfactual([oc(False)] * 4, [oc(False)] * 2 + [oc(True, 1.0)] * 2)
    assert up["verdict"] == "underpowered" and up["route"] is None and "power" in up["why"]
    # a gain in Phi over the target set with no milestone-level success difference is still attributed (bootstrap CI)
    phi = counterfactual([oc(False, 0.0)] * 4, [oc(False, 0.5)] * 4)
    assert phi["verdict"] == "attributed" and phi["dphi_ci"][0] > 0 and phi["phi_route"]
    # one target (binary Phi): 3/4 vs 0/4 is not significant by the exact test, and the bootstrap does not overrule it
    one = counterfactual([oc(False)] * 4, [oc(True, 1.0)] * 3 + [oc(False)])
    assert one["verdict"] == "underpowered" and one["fisher_p"] == pytest.approx(5 / 70, abs=1e-6) and one["dphi_ci"][0] > 0 and not one["phi_route"]
    # masked attempts are dropped (V8), and an arm with none left is unmeasured
    m = counterfactual([oc(False), oc(False, masked=True)], [oc(True, 1.0, masked=True)])
    assert m["verdict"] == "underpowered" and m["masked"] == {"full": 1, "simple": 1}
    strict = counterfactual([oc(False)] * 4, [oc(False)] * 3 + [oc(True, 1.0)], config=CounterfactualConfig(p_ref=0.5))
    assert strict["verdict"] == "underpowered"  # an easier bar for "easy" (p_ref 0.5) no longer calls 1/4 a clear failure


def test_outcome_reads_the_target_with_its_prerequisites() -> None:
    req = {"m01": [], "m02": [], "m03": ["m02"]}
    o = outcome({"m01": True, "m02": False, "m03": True}, req, ["m03"], ["m01", "m03"], phi=1 / 3, reward=1 / 6)
    assert not o.success and o.phi_set == 0.5  # m03 without m02 is not a success; m01 counts
    pre = outcome({"m01": False, "m03": True}, req, ["m03"], ["m01", "m03"])  # m02 pre-applied (absent): counts as passing
    assert pre.success and pre.phi_set == 0.5
    assert outcome(None, req, ["m03"], ["m03"]).masked
    rows = outcomes_of([{"milestones": {"m01": True, "m02": True, "m03": True}, "phi": 1.0, "reward": 1.0, "sample": "s1"},
                        {"masked": True, "milestones": None, "sample": "s2"}],
                       type("T", (), {"milestones": [{"id": k, "requires": v} for k, v in req.items()]})(), ["m03"], ["m03"])
    assert rows[0].success and rows[0].full_pass and rows[1].masked


# ---- 7.5 prerequisite probe, 7.6 stability, band, cause mix ----------------------------------------------------------------
def test_probe_episode_and_prerequisite_probe() -> None:
    calls = [{"function": "repo.read_file", "arguments": {"path": "README"}}, {"function": "repo.shell", "arguments": {"command": "python -m pytest -q"}}]
    ok = probe_episode(calls, "Done.\nPASSED=74", 74)
    assert probe_episode(calls, "PASSED=120", [106, 120])["ok"]  # the repository's own configuration counts doctests too
    tox = [{"function": "repo.shell", "arguments": {"command": "py.test -q --doctest-modules"}, "result": "[exit 0]\n120 passed in 0.4s"}]
    assert probe_episode(tox, "PASSED=120", [106])["ok"] and not probe_episode(tox, "PASSED=150", [106])["ok"]  # a count its own run printed
    assert ok["ok"] and ok["ran_tests"] and ok["reported"] == 74
    assert not probe_episode(calls[:1], "PASSED=74", 74)["ok"]  # never ran a test command
    assert probe_episode(calls, "PASSED=40", 74)["why"].startswith("reported 40")
    assert probe_episode(calls, "PASSED=73", 74)["ok"]  # within tolerance
    assert not probe_episode(calls, "PASSED=3", None)["ok"]  # nothing measured to compare against
    assert probe_episode([{"function": "repo.shell", "arguments": {"command": "python test_parse.py"}}], "PASSED=1", 1)["ok"]
    good = {"ok": True, "passed": 74}
    assert prerequisite_probe(good, ok)["ok"]
    w = prerequisite_probe({"ok": False, "why": "the existing tests do not run through the World's shell"}, None)
    assert not w["ok"] and w["owner"] == "world"
    p = prerequisite_probe(good, probe_episode(calls[:1], "", 74))
    assert not p["ok"] and p["owner"] == "policy"


def test_stability_band_and_cause_mix() -> None:
    orig = [oc(False, phi=x) for x in (0.2, 0.3, 0.3, 0.4)]
    st = stability(orig, [oc(False, phi=x) for x in (0.3, 0.3, 0.2, 0.3)])
    assert st["stable"] and st["pass_ci"][0] == 0.0
    drift = stability(orig, [oc(True, phi=1.0, full=True)] * 4)
    assert drift["stable"] is False and "pass rate" in drift["why"]
    assert stability(orig, [oc(False, masked=True)])["stable"] is None
    assert band([0.0, 0.1, 0.0])["in_band"] and not band([0.0, 0.0, None])["in_band"] and band([0.0, 1.0])["pass_rate"] == 0.5
    assert cause_mix({"skill_gap": 4, "budget": 1, "env_flake": 5})["ok"] is True
    assert cause_mix(["skill_gap", "spec_defect", "spec_defect"])["ok"] is False
    assert cause_mix({"skill_gap": 2})["ok"] is None  # too few labels: unmeasured, never a pass


# ---- S15 -------------------------------------------------------------------------------------------------------------------------
GOOD_PROBE = {"ok": True, "owner": None, "why": ""}
ATTR = {"verdict": "attributed", "why": "x"}


def test_target_verdict_routes() -> None:
    ok = target_verdict(TargetEvidence("t", band={"in_band": True}, cause={"ok": True}, counterfactual=ATTR, probe=GOOD_PROBE,
                                       paraphrase=[{"stable": True}, {"stable": True}]))
    assert ok["ok"] and ok["route"] is None and ok["flags"] == []
    na = target_verdict(TargetEvidence("t", band={"in_band": True}, cause={"ok": True}, counterfactual={"verdict": "not_attributed", "why": "w"},
                                       probe=GOOD_PROBE, paraphrase=[]))
    assert not na["ok"] and na["route"] == "S2" and na["owner"] == "planner"
    up = target_verdict(TargetEvidence("t", band={"in_band": True}, cause={"ok": None, "n": 1}, counterfactual={"verdict": "underpowered", "why": "w"},
                                       probe=GOOD_PROBE, paraphrase=[]))
    assert up["ok"] and any("underpowered" in f for f in up["flags"]) and any("cause mix unmeasured" in f for f in up["flags"])
    drop = target_verdict(TargetEvidence("t", band={"in_band": False, "rewards": [0, 0]}, counterfactual={"verdict": "not_attributed", "why": "w"},
                                         probe={"ok": False, "owner": "world", "why": "tests do not run"}))
    assert drop["route"] == "drop" and drop["owner"] == "world" and len(drop["findings"]) == 3
    para = target_verdict(TargetEvidence("t", band={"in_band": True}, cause={"ok": True}, counterfactual=ATTR, probe=GOOD_PROBE,
                                         paraphrase=[{"stable": True}, {"stable": False, "why": "mean Phi outside the original's CI"}]))
    assert para["route"] == "S2" and para["owner"] == "prompt"
    intent = target_verdict(TargetEvidence("t", band={"in_band": True}, cause={"ok": True}, counterfactual=ATTR,
                                           probe={"ok": False, "owner": "policy", "why": "policy probe: no test command was run"}))
    assert intent["route"] == "S2" and intent["owner"] == "intent"


class Svc:
    def __init__(self, evs: list[TargetEvidence]):
        self.evs, self.results, self.plans = evs, [], 0

    @property
    def evidence(self) -> TargetEvidence:
        return self.evs[min(len(self.results), len(self.evs) - 1)]


def _run(tmp: Path, svc: Svc, rid: str):
    async def s2_plan(ctx):  # a scripted planner: the second plan fixes the task when the evidence list says so
        svc.plans += 1
        return Settle({"plan": {"n": svc.plans, "findings": list(ctx.findings)}})

    assert validate_machine(target_machine(), handlers=target_handlers(s2_plan=s2_plan)) == []
    return asyncio.run(MachineRun(target_machine(), run_dir=tmp / rid, handlers=target_handlers(s2_plan=s2_plan), services=svc, run_id=rid).run())


def test_s15_machine_accepts_loops_back_escalates_and_drops(tmp_path: Path) -> None:
    good = TargetEvidence("t", band={"in_band": True}, cause={"ok": True}, counterfactual=ATTR, probe=GOOD_PROBE, paraphrase=[])
    bad = TargetEvidence("t", band={"in_band": True}, cause={"ok": True}, counterfactual={"verdict": "not_attributed", "why": "w"}, probe=GOOD_PROBE,
                         paraphrase=[])
    res = _run(tmp_path, s := Svc([good]), "ok")
    assert res.status == "done" and len(s.results) == 1
    res = _run(tmp_path, s := Svc([bad, good]), "fixed")  # back to S2 with the findings, then accepted
    assert res.status == "done" and s.plans == 2 and s.results[0]["route"] == "S2"
    res = _run(tmp_path, s := Svc([bad]), "stuck")  # never fixed: the bounded loop escalates (V7), it is never "done"
    assert res.status != "done" and res.status != "rejected" and len(s.results) == 2
    drop = TargetEvidence("t", probe={"ok": False, "owner": "world", "why": "w"})
    res = _run(tmp_path, Svc([drop]), "drop")
    assert res.status == "rejected"


# ---- scripted policies on the toy window ----------------------------------------------------------------------------------------
GAMMA = new_file_patch("pkg/gamma.py", "from pkg.beta import beta\n\n\ndef gamma():\n    return beta() + 1\n")


def _arm(task, patch: str, targets: list[str], compare: list[str], k: int = 4) -> list[Outcome]:
    req = {m["id"]: list(m.get("requires") or []) for m in task.milestones}
    out = []
    for _ in range(k):
        rep = asyncio.run(task.grade_patch(patch, trail=[]))
        out.append(outcome(rep.meta["milestones"], req, targets, compare, phi=rep.meta["phi"], reward=rep.reward))
    return out


def test_counterfactual_with_scripted_policies_on_the_toy_window(built, tmp_path: Path) -> None:
    task, _, _ = built
    assert isinstance(task, Simplifiable) and SIMPLIFY_KINDS["mined_window"] == ("prerequisites", "locate")
    assert task.target_milestones() == ["m03"]
    simple = simplify(task, "prerequisites", targets=["m03"], root=tmp_path)
    compare = [m["id"] for m in simple.milestones]
    # a prerequisite-bound policy: it writes gamma only. On the full task beta is missing, on the simplified one it is there
    full, simp = _arm(task, GAMMA, ["m03"], compare), _arm(simple, GAMMA, ["m03"], compare)
    cf = counterfactual(full, simp)
    assert cf["verdict"] == "attributed" and (cf["success_full"], cf["success_simple"]) == (0, 4)
    # a tool-broken policy (its edits never land): it fails both arms, so the failure is not on the intended axis
    cf2 = counterfactual(_arm(task, "", ["m03"], compare), _arm(simple, "", ["m03"], compare))
    assert cf2["verdict"] == "not_attributed" and cf2["route"] == "S2"
    with pytest.raises(ValueError):
        simplify(task, "baseline_bar")
    # the seeded tooling copy keeps the grader and swaps only the policy's World
    seeded = seeded_tooling(simple, "crucible-lh-p4-seed:x")
    assert seeded.workspace_mode == "ro" and seeded.world_image == "crucible-lh-p4-seed:x" and seeded.sandbox.image == simple.sandbox.image
    assert seeded.milestones == simple.milestones and seeded.prompt == simple.prompt


def test_locate_names_the_target_unit_and_keeps_the_checks(built) -> None:
    task, _, _ = built
    loc = simplify(task, "locate", targets=["m03"])
    sf = loc.window["simplified_from"]
    assert sf["kind"] == "locate" and sf["targets"] == ["m03"] and sf["pre_applied"] == []
    assert sf["hints"][0]["files"] == ["pkg/gamma.py"] and "gamma" in sf["hints"][0]["functions"]
    assert "Where to look: For [R3], the code to change is in `pkg/gamma.py` (look at `gamma`)." in loc.prompt
    assert loc.milestones == task.milestones and loc.check_tests == task.check_tests and loc.base_dir == task.base_dir
    assert loc.ref().simplified["kind"] == "locate" and asyncio.run(ground_truth(loc))["ok"]


# ---- 7.6 paraphrases -------------------------------------------------------------------------------------------------------------
def test_paraphrase_checks_keep_ids_ledger_and_required_names(built) -> None:
    task, _, _ = built
    opening, closing, fixed = split_request(task)
    assert opening == "Release request." and closing == "" and fixed == ""
    para = {"opening": "A new release is due.", "closing": "",
            "requirements": [{"id": "R1", "text": "Calling alpha() gives 1."}, {"id": "R2", "text": "Calling beta() gives 2."},
                             {"id": "R3", "text": "gamma() gives beta() plus one."}]}
    ok = check_paraphrase(task, para)
    assert ok["ok"], ok["findings"]
    assert "[R3] gamma() gives beta() plus one." in ok["prompt"] and [r["milestone"] for r in ok["requirements"]] == ["m01", "m02", "m03"]
    t = paraphrased_task(task, ok, "para1")
    assert t.task_id.endswith("~para1") and t.check_tests == task.check_tests and t.prompt == ok["prompt"]
    # ``alpha`` is a name the hidden tests import from a module A has (``windows.required_names``): it must stay stated
    dropped = check_paraphrase(task, {**para, "requirements": [{"id": "R1", "text": "The first helper gives 1."}, *para["requirements"][1:]]})
    assert not dropped["ok"] and any("required name no longer stated: alpha" in f for f in dropped["findings"])
    renum = check_paraphrase(task, {**para, "requirements": para["requirements"][:2]})
    assert not renum["ok"] and renum["findings"][0].startswith("requirement ids changed")
