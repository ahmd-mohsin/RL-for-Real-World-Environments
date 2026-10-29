"""crucible.synth.meta.update: the regularised update rule R1–R10 (part2-paper-core §9.4; M3 acceptance).

A toy Θ whose F component's data *is* its measured behaviour (skill, per-domain gain, recall, cells, cost...), and a
seeded evaluator with per-unit noise: one genuine improvement is accepted, and one planted update per regulariser is
refused, naming that regulariser.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from crucible.synth.meta.update import (
    Evidence,
    Ledger,
    Measurement,
    Theta,
    ThetaStore,
    Update,
    UpdateConfig,
    UpdateEngine,
    View,
    edit_budget,
    entropy,
    kl,
    lcb,
    leakage_critic,
)

DOMAINS = ["ap-test", "desk-test"]
BASE_F = {"skill": 0.0, "domain_gain": {}, "fr": {"D1": 0.8, "D2": 0.9}, "fa": 0.0, "archive_ok": True, "p3": 0.8, "p4": 0.5,
          "cells": {"hold": 10, "approve": 10, "resolve": 10, "priority": 10}, "cost": 1.0, "items": {"op_a": {}}, "item_gain": {}}


class Toy:
    """J per unit = 1 + skill + domain gain + seeded noise (the same noise for every Θ on one view, so pairing cancels it).
    The control (a budget above Θ's own cost) buys ``budget_value`` per extra dollar. ``drift`` flips skill's sign
    after the given round: an accepted Θ that later regresses."""

    def __init__(self, budget_value: float = 0.0, units: int = 8, noise: float = 0.01):
        self.budget_value, self.units, self.noise = budget_value, units, noise
        self.drift_after: int | None = None
        self.calls = 0

    async def __call__(self, theta: Theta, view: View, budget: float | None) -> Measurement:
        self.calls += 1
        f = theta.data["F"]
        skill = f["skill"]
        if self.drift_after is not None and view.round > self.drift_after:
            skill = -skill
        rng = random.Random(f"toy/{view.hash}")
        js, doms = [], []
        extra = self.budget_value * max(0.0, (budget or f["cost"]) - f["cost"])
        for d in view.domains:
            for _ in range(self.units):
                js.append(1.0 + skill + f["domain_gain"].get(d, 0.0) + extra + rng.uniform(-self.noise, self.noise))
                doms.append(d)
        return Measurement(j_units=js, unit_domains=doms, fr=dict(f["fr"]), fa=f["fa"], archive_ok=f["archive_ok"], p3=f["p3"], p4=f["p4"],
                           cells=dict(f["cells"]), cost=f["cost"] if budget is None else budget, item_gain=dict(f.get("item_gain", {})))


class Sampler:
    """Fresh views by default; ``reuse``/``locked``/``planters`` plant an R1 violation into the next view."""

    def __init__(self):
        self.n = 0
        self.reuse: str | None = None
        self.locked = False
        self.planters: list[str] | None = None

    async def __call__(self, round: int, purpose: str, used: frozenset[str]) -> View:
        self.n += 1
        h = self.reuse or f"view-{self.n}"
        planters = self.planters if self.planters is not None else [f"P@{i}" for i in range(1, 10)]
        return View(hash=h, round=round, domains=list(DOMAINS), planters=planters, locked=self.locked, seed=self.n)


def _engine(tmp_path=None, **cfg) -> tuple[UpdateEngine, Toy, Sampler]:
    store = ThetaStore(tmp_path / "theta" if tmp_path is not None else None)
    store.init("F", BASE_F)
    store.init("P", {"planters": ["dev"]})
    toy, s = Toy(budget_value=cfg.pop("budget_value", 0.0)), Sampler()
    ledger = Ledger(tmp_path / "ledger.jsonl" if tmp_path is not None else None)
    eng = UpdateEngine(store, toy, s, config=UpdateConfig(rounds=5, **cfg), ledger=ledger, anchor_tokens=["INV-1003"])
    return eng, toy, s


def _f(**changes) -> dict:
    d = json.loads(json.dumps(BASE_F))
    for k, v in changes.items():
        d[k] = v
    return d


def _u(uid: str = "u1", edits: int = 1, domains=("ap-test",), **changes) -> Update:
    return Update(id=uid, component="F", data=_f(**changes), edits=[f"e{i}" for i in range(edits)], author_family="kimi",
                  evidence=Evidence(domains=list(domains), task_ids=["ap/hold-1003"], entity_keys=["103"], builder_families=["glm"]))


GOOD = {"skill": 0.1}


async def test_a_genuine_improvement_is_accepted_and_ledgered(tmp_path):
    eng, _toy, _ = _engine(tmp_path)
    fl = await eng.calibrate()
    assert fl.delta > 0 and len(fl.j_values) == 3 and len(fl.views) == 3  # measured from 3 fresh views of the unchanged Θ
    d = await eng.decide(_u(**GOOD), round=0)
    assert d.accepted and d.failed == [] and d.to_version == 2 and eng.store.get("F")["skill"] == 0.1
    assert d.metrics["lcb"] > fl.delta and d.metrics["j_new"] > d.metrics["j_star"]
    recs = eng.ledger.records()
    assert [r.kind for r in recs] == ["calibrate", "update"] and recs[1].view == d.view
    # the store is durable and versioned; the ledger is hash-chained
    again = ThetaStore(tmp_path / "theta")
    assert again.head("F") == 2 and again.get("F", 1)["skill"] == 0.0
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    rec = json.loads(lines[1])
    rec["rec"]["accepted"] = False
    (tmp_path / "ledger.jsonl").write_text("\n".join([lines[0], json.dumps(rec)]) + "\n")
    with pytest.raises(ValueError):
        eng.ledger.records()


async def test_r1_a_reused_or_locked_view_is_refused():
    eng, _toy, s = _engine()
    await eng.calibrate()
    s.reuse = "view-1"  # a calibration view
    d = await eng.decide(_u(**GOOD), round=0)
    assert not d.accepted and d.rules == ["R1"] and not d.evaluated
    s.reuse, s.locked = None, True
    d = await eng.decide(_u("u2", **GOOD), round=0)
    assert d.rules == ["R1"] and "locked" in d.failed[0].detail
    s.locked, s.planters = False, ["P@2"]  # a planted batch drawn without the first planter version
    d = await eng.decide(_u("u3", **GOOD), round=0)
    assert d.rules == ["R1"] and "P@1" in d.failed[0].detail
    s.planters = None
    assert (await eng.decide(_u("u4", **GOOD), round=0)).accepted


async def test_r2_a_gain_inside_the_noise_floor_is_refused_and_the_floor_is_never_tuned():
    eng, toy, _ = _engine()
    with pytest.raises(RuntimeError):
        await eng.decide(_u(**GOOD), round=0)  # no decision before the floor is measured
    fl = await eng.calibrate()
    d = await eng.decide(_u(skill=fl.delta / 4), round=0)
    assert not d.accepted and "R2" in d.rules and d.metrics["lcb"] <= fl.delta
    with pytest.raises(PermissionError):
        eng.set_floor(0.0)
    with pytest.raises(ValueError):
        await eng.calibrate(repeats=2)
    toy.noise = 0.0  # a quieter recalibration cannot lower the floor
    assert (await eng.calibrate()).delta == fl.delta


async def test_r3_a_gain_that_forgets_a_defect_class_is_refused():
    eng, _, _ = _engine()
    await eng.calibrate()
    d = await eng.decide(_u(fr={"D1": 0.8, "D2": 0.6}, **GOOD), round=0)
    assert d.rules == ["R3"] and "FR_D2" in d.failed[0].detail
    d = await eng.decide(_u("u2", archive_ok=False, **GOOD), round=0)
    assert d.rules == ["R3"] and "archived" in d.failed[0].detail
    d = await eng.decide(_u("u3", fa=0.2, **GOOD), round=0)
    assert d.rules == ["R3"] and "FA" in d.failed[0].detail
    d = await eng.decide(_u("u4", p3=0.5, **GOOD), round=0)
    assert d.rules == ["R3"] and "p3" in d.failed[0].detail


async def test_r4_certification_goodhart_is_refused():
    eng, _, _ = _engine(kappa=0.05, h_min=1.0)
    await eng.calibrate()
    easy = {"hold": 1, "approve": 1, "resolve": 1, "priority": 1, "read_only_lookup": 60}  # drift to tasks that are easy to certify
    d = await eng.decide(_u(cells=easy, **GOOD), round=0)
    assert d.rules == ["R4"] and "KL" in d.failed[0].detail and "H(q')" in d.failed[1].detail
    assert kl(easy, BASE_F["cells"]) > 0.05 > kl(BASE_F["cells"], BASE_F["cells"]) and entropy(easy) < 1.0 < entropy(BASE_F["cells"])


async def test_r5_edit_budget_cost_bar_and_pruning():
    assert [edit_budget(r, 5) for r in range(6)] == [3, 3, 2, 1, 1, 1]
    eng, _, _ = _engine(beta=0.05)
    await eng.calibrate()
    d = await eng.decide(_u(edits=3, **GOOD), round=4)  # three edits when the budget has annealed to one
    assert d.rules == ["R5"] and not d.evaluated
    d = await eng.decide(_u("u2", cost=5.0, **GOOD), round=0)  # costs 4 more probe-dollars, buys no recall
    assert d.rules == ["R5"] and "per probe-dollar" in d.failed[0].detail
    assert (await eng.decide(_u("u3", edits=3, **GOOD), round=0)).accepted  # three edits are fine at round 0
    # an item with no marginal gain over W rounds is pruned (through a fresh view and non-regression)
    assert await eng.prune("F", round=1, remove=lambda d, items: d) is None  # nothing stale yet
    eng.last_gain["op_a"] = 0
    p = await eng.prune("F", round=3, remove=lambda d, items: {**d, "items": {k: v for k, v in d["items"].items() if k not in items}})
    assert p.accepted and p.kind == "prune" and eng.store.get("F")["items"] == {} and "op_a" not in eng.last_gain


async def test_r6_leakage_and_family_separation_are_refused():
    eng, _, _ = _engine()
    await eng.calibrate()
    u = _u(**GOOD).model_copy(update={"diff": "+ when the task is ap/hold-1003, set reason to price_variance"})
    d = await eng.decide(u, round=0)
    assert d.rules == ["R6"] and "task_id:ap/hold-1003" in d.failed[0].detail and not d.evaluated
    u = _u("u2", **GOOD).model_copy(update={"diff": "+ prefer invoice 103"})
    assert (await eng.decide(u, round=0)).rules == ["R6"]  # an entity key
    u = _u("u3", **GOOD).model_copy(update={"diff": "+ look for INV-1003 style numbers"})
    assert "anchor:INV-1003" in (await eng.decide(u, round=0)).failed[0].detail  # anchor content
    u = _u("u4", **GOOD).model_copy(update={"author_family": "glm"})
    d = await eng.decide(u, round=0)
    assert d.rules == ["R6"] and "built the source tasks" in d.failed[0].detail
    assert leakage_critic(_u("u5", **GOOD).model_copy(update={"diff": "+ swap the entity key for a sibling of the same collection"})) == []

    async def other_family(u: Update) -> list[str]:  # an injected critic (M4: a different family reads the diff)
        return ["memorises an anchor"] if "anchor" in u.diff else []

    eng2, _, _ = _engine()
    eng2.critic = other_family
    await eng2.calibrate()
    assert (await eng2.decide(_u(**GOOD).model_copy(update={"diff": "anchor"}), round=0)).rules == ["R6"]


async def test_r7_a_gain_only_on_the_evidence_domain_is_refused():
    eng, _, _ = _engine()
    await eng.calibrate()
    d = await eng.decide(_u(domain_gain={"ap-test": 0.2}), round=0)  # J rises overall, all of it on the domain that produced the evidence
    assert d.rules == ["R7"] and "held-out-domain gain" in d.failed[0].detail
    d = await eng.decide(_u("u2", domain_gain={"desk-test": 0.2}), round=0)  # the same gain on the domain the evidence never saw
    assert d.accepted
    d = await eng.decide(_u("u3", domains=DOMAINS, skill=0.2), round=0)  # evidence from every domain the view has: nothing to transfer to
    assert d.rules == ["R7"] and "no domain outside" in d.failed[0].detail


async def test_r8_judge_only_rounds_are_capped():
    eng, _, _ = _engine(judge_round_cap=2)
    await eng.calibrate()

    def judged(uid: str, skill: float, executed: int = 0) -> Update:
        u = _u(uid, skill=skill)
        return u.model_copy(update={"evidence": u.evidence.model_copy(update={"kind": "judge", "new_executed": executed})})

    assert (await eng.decide(judged("j1", 0.1), round=0)).accepted
    assert (await eng.decide(judged("j2", 0.2), round=1)).accepted
    d = await eng.decide(judged("j3", 0.3), round=2)
    assert d.rules == ["R8"] and not d.evaluated
    assert (await eng.decide(judged("j4", 0.3, executed=2), round=2)).accepted  # new executed counterexamples lift the cap


async def test_r9_a_gain_that_more_budget_would_buy_is_refused():
    eng, _, _ = _engine(budget_value=0.1)
    await eng.calibrate()
    d = await eng.decide(_u(cost=3.0, **GOOD), round=0)  # +0.1 J for +2 dollars; the incumbent with 2 more dollars gets +0.2
    assert d.rules == ["R9"] and d.metrics["j_control"] > d.metrics["j_new"]
    assert (await eng.decide(_u("u2", cost=1.0, **GOOD), round=0)).accepted  # the same gain at the same spend


async def test_r10_irreversible_updates_are_refused_and_regressions_roll_back(tmp_path):
    eng, toy, _ = _engine(tmp_path)
    await eng.calibrate()
    d = await eng.decide(_u(**GOOD).model_copy(update={"data": {**BASE_F, "skill": math.nan}}), round=0)
    assert d.rules == ["R10"] and not d.evaluated  # not restorable by a plain write
    d = await eng.decide(_u("u2").model_copy(update={"data": {**BASE_F, "op": object()}}), round=0)
    assert d.rules == ["R10"]
    ok = await eng.decide(_u("u3", **GOOD), round=0)
    assert ok.accepted and eng.store.head("F") == 2
    assert (await eng.audit(round=1)).metrics["regressed"] == []  # no regression: nothing rolled back
    assert eng.store.head("F") == 2
    toy.drift_after = 1  # the accepted Θ turns out worse on later fresh views
    rb = await eng.audit(round=2)
    assert rb.accepted and rb.kind == "rollback" and eng.store.head("F") == 1 and eng.store.get("F") == BASE_F
    assert ThetaStore(tmp_path / "theta").head("F") == 1  # the rollback is a plain write of the head
    # a version that no longer restores exactly is caught before the head moves
    (tmp_path / "theta" / "F" / "v1.json").write_text(json.dumps({**BASE_F, "skill": 9}))
    toy.drift_after = None
    d = await eng.decide(_u("u4", **GOOD), round=3)
    assert d.rules == ["R10"] and eng.store.head("F") == 1


def test_lcb_is_paired_when_it_can_be():
    a = [1.0 + i * 0.1 for i in range(10)]
    b = [x - 0.05 for x in a]
    assert lcb(a, b) == pytest.approx(0.05)  # paired: the unit spread cancels
    assert lcb(a, b[:9]) < 0  # unpaired (Welch): it does not
