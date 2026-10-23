"""The difficulty subsystem (swarm-pipeline/12): online p̂ (``forge.difficulty``), the sampler and reverse curriculum
(``train.curriculum``), the controller, predictor, checkpoint-conditioned difficulty and checkpoint starts
(``factory.difficulty``), the W&B views (``observe.difficulty``), and the Tinker trainer's ``sampler="difficulty"``."""

from __future__ import annotations

import asyncio
import json
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from crucible.core.schema.rollout import RolloutRecord
from crucible.credit.compose import Episode, turn_credit
from crucible.factory import checkpoint as ck
from crucible.factory.difficulty import (
    HeldOutRefused,
    Journal,
    check_start,
    checkpoint_levels,
    checkpoint_start,
    conditional,
    control,
    features,
    fit,
    gate_start,
    loo,
    plan,
    training_rows,
    uncertainty_reward,
)
from crucible.forge.difficulty import DifficultyTracker, Draw
from crucible.forge.learnability import estimate
from crucible.observe import difficulty as views
from crucible.train.bridges.tinker.fake import FakeTinker
from crucible.train.bridges.tinker.trainer import Group, TinkerTrainer, TrainConfig
from crucible.train.chat import ByteChatCodec
from crucible.train.curriculum import DifficultySampler, SamplerConfig, draws_of, requires_of
from crucible.train.quarantine import QuarantineConfig, TaskQuarantine

MODEL = "Qwen/Qwen3.6-35B-A3B"
# m3 needs m2 needs m1; m4 independent
MS = [{"id": "m1", "requires": []}, {"id": "m2", "requires": ["m1"]}, {"id": "m3", "requires": ["m2"]}, {"id": "m4", "requires": []}]
REQ = requires_of(MS)


# ---- online p̂ ------------------------------------------------------------------------------------------------------------
def test_masked_draws_are_never_evidence_and_budget_stops_are_not_skill() -> None:
    tr = DifficultyTracker(decay=1.0)
    r = tr.observe("t", 0, [Draw(True, 1.0), Draw(False, 0.5), Draw(None, 0.9), Draw(False, 0.25, stop="max_steps")])
    assert r.full.n == 2 and r.p == 0.5 and r.masked == 1 and r.budget == 1
    assert r.phi_n == 3  # the budget stop's measured progress is real; the masked draw's Φ is not counted
    tr2 = DifficultyTracker(decay=1.0)
    r2 = tr2.observe("t", 0, [Draw(True, 1.0), Draw(False, 0.5), Draw(None, 0.0), Draw(None, 1.0)])
    assert (r2.full.k, r2.full.n) == (r.full.k, r.full.n)


def test_old_evidence_decays_as_the_policy_moves() -> None:
    tr = DifficultyTracker(decay=0.5)
    tr.observe("t", 0, [Draw(False, 0.0)] * 8)
    r = tr.observe("t", 3, [Draw(True, 1.0)] * 2)
    assert r.full.n == pytest.approx(8 * 0.125 + 2) and r.p == pytest.approx(2 / 3)


def test_phi_spread_keeps_a_zero_pass_task_learnable() -> None:
    tr = DifficultyTracker(decay=1.0)
    flat = tr.observe("flat", 0, [Draw(False, 0.0)] * 4)
    spread = tr.observe("spread", 0, [Draw(False, x) for x in (0.0, 0.25, 0.5, 0.75)])
    assert flat.signal() == 0.0 and spread.signal() == pytest.approx(4 * statistics.pvariance([0.0, 0.25, 0.5, 0.75]))
    assert tr.rate("new").signal() == 1.0  # never measured: worth measuring


def test_per_milestone_rates_use_the_c1_rule_and_condition_on_prerequisites() -> None:
    tr = DifficultyTracker(decay=1.0, requires={"t": REQ})
    draws = [Draw(False, 0.25, milestones={"m1": True, "m2": False, "m3": False, "m4": False}),
             Draw(False, 0.5, milestones={"m1": True, "m2": True, "m3": False, "m4": False}),
             Draw(False, 0.25, milestones={"m1": False, "m2": True, "m3": True, "m4": False})]  # m2/m3 out of order: not credited
    r = tr.observe("t", 0, draws)
    assert r.milestone["m2"].p == pytest.approx(1 / 3)
    assert r.conditional["m2"].p == pytest.approx(1 / 2)  # given m1 credited: 1 of 2
    assert r.conditional["m3"].p == 0.0 and r.conditional["m3"].n == 1


def test_tracker_round_trips() -> None:
    tr = DifficultyTracker(decay=0.8, requires={"t": REQ})
    tr.observe("t", 2, [Draw(True, 1.0, milestones={"m1": True, "m2": True, "m3": True, "m4": True}), Draw(False, 0.5)])
    back = DifficultyTracker(decay=0.8).load(json.loads(json.dumps(tr.to_json())))
    assert back.rows() == tr.rows()


# ---- the sampler ---------------------------------------------------------------------------------------------------------
def test_the_sampler_refuses_held_out_tasks() -> None:
    with pytest.raises(ValueError, match="held-out"):
        DifficultySampler(["a", "h"], splits={"a": "train", "h": "heldout"})


def test_out_of_band_tasks_are_filtered_and_in_band_ones_weighted() -> None:
    s = DifficultySampler(["easy", "mid", "dead"], SamplerConfig(explore=0.0, min_n=4), seed=1)
    s.observe("easy", [Draw(True, 1.0)] * 40, 0)  # 40 draws: the interval must exclude the band (8 cannot)
    s.observe("mid", [Draw(True, 1.0)] * 4 + [Draw(False, 0.0)] * 4, 0)
    s.observe("dead", [Draw(False, 0.0)] * 40, 0)
    picks =[s.tasks[s.pick(0, 1)] for _ in range(50)]
    assert set(picks) == {"mid"}


def test_explore_re_measures_out_of_band_tasks() -> None:
    s = DifficultySampler(["mid", "dead"], SamplerConfig(explore=1.0, min_n=4), seed=3)
    s.observe("mid", [Draw(True, 1.0)] * 4 + [Draw(False, 0.0)] * 4, 0)
    s.observe("dead", [Draw(False, 0.0)] * 8, 0)
    assert "dead" in {s.tasks[s.pick(0, 1)] for _ in range(200)}


def test_a_quarantined_task_is_not_sampled() -> None:
    q = TaskQuarantine(["a", "b"], QuarantineConfig(q=1, r=5))
    s = DifficultySampler(["a", "b"], SamplerConfig(explore=0.0), quarantine=q, seed=0)
    s.observe("a", [Draw(False, 0.0)] * 4, 0, rewards=[0.0] * 4, phis=[0.0] * 4)  # flat: quarantined for 5 steps
    assert q.blocked("a", 1)
    assert {s.tasks[s.pick(0, 1)] for _ in range(30)} == {"b"}


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 3), st.lists(st.sampled_from([True, False, None]), min_size=1, max_size=6)), max_size=25), st.integers(0, 10**6))
def test_property_never_samples_held_out_and_never_weights_a_masked_draw(obs, seed) -> None:
    tasks = ["t0", "t1", "t2", "t3"]
    s = DifficultySampler(tasks, splits=dict.fromkeys(tasks, "train"), seed=seed)
    shadow = DifficultyTracker(decay=s.config.decay)
    for step, (i, outs) in enumerate(obs):
        draws = [Draw(o, None if o is None else float(o)) for o in outs]
        s.observe(tasks[i], draws, step)
        shadow.observe(tasks[i], step, [d for d in draws if not d.masked])  # the same evidence with the masked draws removed
        assert s.tasks[s.pick(step, step)] in tasks
    for t, r in s.tracker.rates.items():
        assert (round(r.full.k, 9), round(r.full.n, 9)) == (round(shadow.rate(t).full.k, 9), round(shadow.rate(t).full.n, 9))
    with pytest.raises(ValueError):
        DifficultySampler([*tasks, "h"], splits={**dict.fromkeys(tasks, "train"), "h": "heldout"}, seed=seed)


def test_reverse_curriculum_moves_the_start_earlier_as_p_rises() -> None:
    levels = {"t@2": ("t", 2), "t@1": ("t", 1), "t": ("t", 0)}
    s = DifficultySampler(list(levels), SamplerConfig(explore=0.0, min_n=4), levels=levels, seed=0)
    assert {s.tasks[s.pick(0, 0)] for _ in range(20)} == {"t@2"}  # starts late
    s.observe("t@2", [Draw(True, 1.0)] * 8, 1)
    assert s.current["t"] == 1 and s.events[-1]["event"] == "start_earlier"
    assert {s.tasks[s.pick(0, 2)] for _ in range(20)} == {"t@1"}
    s.observe("t@1", [Draw(False, 0.0)] * 40, 3)
    assert s.current["t"] == 2 and s.events[-1]["event"] == "start_later"


def test_sampler_state_round_trips_and_resumes_the_same_picks() -> None:
    a = DifficultySampler(["x", "y", "z"], seed=7)
    a.observe("x", [Draw(True, 1.0), Draw(False, 0.0)] * 3, 0)
    doc = json.loads(json.dumps(a.to_json()))
    b = DifficultySampler(["x", "y", "z"], seed=0).load(doc)
    assert [a.pick(0, 1) for _ in range(10)] == [b.pick(0, 1) for _ in range(10)]


def test_draws_of_rollout_records() -> None:
    recs = [RolloutRecord.model_validate({"rollout_id": f"r{j}", "sample": "s", "task": "t", "harness": "h", "policy": "p", **kw})
            for j, kw in enumerate([{"reward": 1.0, "gates": {"c_m1": True}, "progress": {"phi_final": 1.0}},
                                    {"reward": 0.25, "gates": {"c_m1": False}, "stop": "max_steps", "progress": {"phi_final": 0.5}},
                                    {"reward": None, "mask_sample": True}])]
    d = draws_of(recs)
    assert [x.passed for x in d] == [True, False, None] and d[1].budget and d[2].masked


# ---- the trainer with sampler="difficulty" -------------------------------------------------------------------------------
def _env(tid: str, split: str = "train") -> SimpleNamespace:
    return SimpleNamespace(task=SimpleNamespace(task_id=tid, split=split, verifier_version="v1", milestones=[]), run_spec=None, policy_kwargs=dict)


class ScriptedTrainer(TinkerTrainer):
    script: ClassVar[dict[str, tuple[list[float], list[float]]]] = {}

    async def _group(self, env, gi):
        tid = str(env.task.task_id)
        rs, ps = self.script[tid]
        recs = [RolloutRecord.model_validate({"rollout_id": f"{tid}-{gi}-{j}", "sample": f"s{j}", "task": tid, "harness": "h", "policy": "p", "reward": r,
                                              "progress": {"phi_final": p}}) for j, (r, p) in enumerate(zip(rs, ps, strict=True))]
        return Group(gi, env, tid, "v1", f"g{gi}", recs, self.state["version"], 0.0)

    async def _step(self, batch):
        step = self.state["step"] + 1
        self.state.update(step=step, version=step)
        self.step_rows.append({"step": step, "tasks": [g.task_id for g in batch], "groups": [g.run_id for g in batch]})
        self.tracking.log({"reward/mean": statistics.fmean(x for g in batch for x in g.rewards())}, step)


def test_the_tinker_trainer_samples_by_difficulty_and_saves_its_state(tmp_path: Path) -> None:
    cfg = TrainConfig(model=MODEL, group_size=4, groups_per_step=1, steps=12, require_band=False, drop_zero_spread=False, sampler="difficulty",
                      explore=0.0, quarantine_q=0)
    t = ScriptedTrainer([_env("solved"), _env("mid")], config=cfg, backend=FakeTinker(), run_dir=tmp_path / "run", codec=ByteChatCodec(), objective_id="o")
    t.script = {"solved": ([1.0] * 4, [1.0] * 4), "mid": ([1.0, 0.0, 1.0, 0.0], [1.0, 0.5, 1.0, 0.25])}
    out = asyncio.run(t.run())
    tail = [ts for s in out["steps_log"][6:] for ts in s["tasks"]]
    assert tail and set(tail) == {"mid"}  # once measured, the solved task (p = 1, no Φ spread) is out of band
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert "difficulty" in state and out["difficulty"]["sampler"] == "difficulty"
    with pytest.raises(ValueError):
        TrainConfig(model=MODEL, sampler="nope")


def test_switching_the_sampler_on_resume_is_allowed(tmp_path: Path) -> None:
    kw = {"backend": FakeTinker(), "run_dir": tmp_path / "r", "codec": ByteChatCodec(), "objective_id": "o"}
    ScriptedTrainer([_env("a")], config=TrainConfig(model=MODEL, require_band=False), **kw)._open()
    ScriptedTrainer([_env("a")], config=TrainConfig(model=MODEL, require_band=False, sampler="difficulty", band_lo=0.2), **kw)._open()


# ---- the controller ------------------------------------------------------------------------------------------------------
def _rate(k: int, n: int):
    return estimate([True] * k + [False] * (n - k), model=MODEL, harness="h")


def test_the_controller_refuses_held_out_tasks() -> None:
    with pytest.raises(HeldOutRefused):
        control("h", _rate(0, 8), split="heldout")


def test_controller_actions_follow_the_band_and_the_causes() -> None:
    assert control("t", _rate(4, 8)).action == "keep"
    assert control("t", _rate(0, 0)).action == "sample_more"
    assert control("t", _rate(0, 8), causes={"budget": 6, "skill_gap": 2}).action == "budget_bound"
    assert control("t", _rate(0, 8), causes={"spec_defect": 5, "skill_gap": 3}).action == "fix_task"
    assert control("t", _rate(0, 8), causes={"skill_gap": 3, "other": 5}).action == "fix_task"
    assert control("t", _rate(0, 8), frontier_passes=False, causes={"skill_gap": 8}).action == "fix_task"
    h = control("t", _rate(8, 8), applied=["K4"])
    assert h.action == "harden" and h.knobs[0] == "H2" and "K4" in h.reason
    e = control("t", _rate(0, 8), causes={"skill_gap": 8}, modes={"claimed_done_without_progress/reproduce_from_report": 5})
    assert e.action == "ease" and e.knobs[0] == "K4" and "C1" not in e.knobs[:1]
    c = control("t", _rate(0, 8), causes={"skill_gap": 8}, modes={"stalled": 4}, conditional_rates={"m4": 0.5})
    assert c.knobs[0] == "C1" and "checkpoint start" in c.reason
    assert all(k not in control("t", _rate(0, 8), causes={"skill_gap": 8}, applied=["K1", "K2"]).knobs for k in ("K1", "K2"))
    assert uncertainty_reward(0.5) == 1.0 and uncertainty_reward(0.0) == 0.0 and uncertainty_reward(1.0) == 0.0


def test_the_journal_records_every_action_with_its_reason(tmp_path: Path) -> None:
    j = Journal(tmp_path / "difficulty_actions.jsonl")
    j.add(control("t", _rate(0, 8), causes={"skill_gap": 8}), step=3)
    row = j.rows()[0]
    assert row["action"] == "ease" and row["reason"] and row["step"] == 3 and row["knob_text"]


# ---- checkpoint-conditioned difficulty and checkpoint starts ---------------------------------------------------------------
def test_conditional_difficulty_separates_full_task_and_checkpoint_conditioned_rates() -> None:
    eps = [{"milestones": {"m1": True, "m2": True, "m3": False, "m4": True}, "passed": False},
           {"milestones": {"m1": True, "m2": False, "m3": False, "m4": True}, "passed": False},
           {"milestones": {"m1": True, "m2": True, "m3": True, "m4": True}, "passed": True},
           {"milestones": {"m1": False, "m2": False, "m3": False, "m4": False}, "passed": False, "stop": "max_steps"},  # budget
           {"masked": True}]
    c = conditional(eps, MS)
    assert c["n"] == 3 and c["budget"] == 1 and c["masked"] == 1 and c["p_full"] == pytest.approx(1 / 3)
    assert c["milestones"]["m3"]["p"] == pytest.approx(1 / 3) and c["milestones"]["m3"]["p_given_prereqs"] == pytest.approx(1 / 2)
    assert c["phi_mean"] == pytest.approx(statistics.fmean([0.75, 0.5, 1.0]))


def test_checkpoint_levels_are_prerequisite_closed_and_leave_work() -> None:
    lv = checkpoint_levels(MS)
    assert lv == [["m1"], ["m1", "m4"], ["m1", "m4", "m2"]]
    req = {m["id"]: m["requires"] for m in MS}
    assert all(all(r in pre for m in pre for r in req[m]) for pre in lv)


class FakeTask:
    def __init__(self, split: str = "train"):
        self.task_id, self.split = "t", split
        self.milestones = [{**m, "unit": f"u{m['id']}"} for m in MS]

    def simplify_units(self, units, *, kind, root=None):
        pre = {m["id"] for m in self.milestones if m["unit"] in units}
        v = SimpleNamespace(task_id=f"t~chec-{len(pre)}", kind=kind, units=set(units),
                            milestones=[{**m, "requires": [r for r in m["requires"] if r not in pre]} for m in self.milestones if m["id"] not in pre])
        return v


def test_a_checkpoint_start_is_built_by_simplify_units_and_tagged_for_the_curriculum() -> None:
    v = checkpoint_start(FakeTask(), ["m1", "m2"], level=2)
    assert v.kind == "checkpoint" and v.units == {"um1", "um2"} and v.curriculum == ("t", 2)
    assert [m["id"] for m in v.milestones] == ["m3", "m4"] and v.checkpoint_of["pre_applied"] == ["m1", "m2"]
    with pytest.raises(HeldOutRefused):
        checkpoint_start(FakeTask("heldout"), ["m1"], level=1)


def test_the_start_check_and_the_gates_must_both_pass() -> None:
    good = {"m1": True, "m2": True, "m3": False, "m4": False}
    assert check_start(MS, ["m1", "m2"], good) == []
    assert check_start(MS, ["m1", "m2"], {**good, "m2": False})  # a pre-applied fix does not hold
    assert check_start(MS, ["m1", "m2"], {**good, "m4": True})  # nothing left for m4
    assert check_start(MS, ["m1", "m2"], {**good, "m3": None})  # unmeasured is a problem

    async def status(_v):
        return good

    async def gates(_v):
        return {"ok": True, "passed": ["S5", "S8"]}

    out = asyncio.run(gate_start(SimpleNamespace(task_id="v"), original_milestones=MS, pre=["m1", "m2"], status=status, gates=gates))
    assert out["ok"] and out["start_problems"] == []


def test_a_checkpoint_starts_prerequisites_never_earn_credit() -> None:
    pre = ["m1", "m2"]
    w = ck.uniform_weights(MS)
    start = {"m1": True, "m2": True, "m3": False, "m4": False}
    # (a) graded on the ORIGINAL milestones: Φ0 is the baseline; the pre-applied milestones are credited at step 0 only
    cps = [(5, ["periodic"], {"m1": True, "m2": True, "m3": False, "m4": True}), (9, ["commit"], {"m1": True, "m2": True, "m3": True, "m4": True})]
    phi0, pts = ck.series(cps, MS, w, start=start)
    rows = ck.progress_rows(phi0, pts, [(s, "a", f"c{s}") for s in range(1, 11)], start_credited=ck.credited(start, MS))
    assert phi0 == pytest.approx(0.5)
    assert all(not set(pre) & set(r["gained"]) for r in rows if r["step"] != 0)
    assert sum(r["delta_phi"] for r in rows) == pytest.approx(1.0 - phi0)  # only progress made after the start
    # an episode that only keeps the pre-applied fixes earns no ΔΦ at all, so no positive turn credit
    idle = Episode(reward=0.25, phis=[phi0] * 10, phi0=phi0)
    worked = Episode(reward=1.0, phis=[r["phi"] for r in rows if r["step"] > 0], phi0=phi0)
    credit = turn_credit([idle, worked], eta=0.5)
    assert all(abs(x) < 1e-12 for x in credit[0])
    # (b) the variant itself: the pre-applied milestones are gone from Φ, so no state can credit them
    v = checkpoint_start(FakeTask(), pre, level=2)
    vw = ck.uniform_weights(v.milestones)
    full = {"m1": True, "m2": True, "m3": True, "m4": True}
    assert not set(pre) & ck.credited(full, v.milestones) and ck.potential(full, v.milestones, vw) == pytest.approx(1.0)
    assert ck.potential({"m1": True, "m2": True}, v.milestones, vw) == 0.0


# ---- the predictor -------------------------------------------------------------------------------------------------------
def _cand(faults: int, lang: str = "go", tid: str = "x") -> dict:
    return {"id": tid, "faults": faults, "edges": faults - 1, "modules": 1, "oracle_lines": 40 * faults, "milestone_tests": 3 * faults, "lang": lang}


def test_the_predictor_learns_that_more_faults_are_harder_and_reports_loo_against_a_baseline() -> None:
    rows = []
    for i in range(12):
        f = 2 + i % 3
        k = {2: 6, 3: 3, 4: 0}[f]
        rows.append((f"t{i}", features(_cand(f)), k, 8))
    m = fit([(f, k, n) for _, f, k, n in rows])
    assert m.p(features(_cand(2))) > m.p(features(_cand(3))) > m.p(features(_cand(4)))
    rep = loo(rows)
    assert rep["tasks"] == 12 and rep["mae"] is not None and rep["mae"] < rep["mae_baseline"]
    pr = m.predict(features(_cand(3)), probe=(2, 2))
    assert pr["p"] > pr["p_static"] and pr["advisory"]


def test_plan_orders_by_learnability_and_only_suggests_skips() -> None:
    out = plan({"a": {"p": 0.5, "lo": 0.3, "hi": 0.7}, "b": {"p": 0.01, "lo": 0.0, "hi": 0.015}, "c": {"p": 0.2, "lo": 0.1, "hi": 0.3}})
    assert [r["task"] for r in out] == ["a", "c", "b"] and out[-1]["suggest_skip"] and all(r["advisory"] for r in out)


def test_training_rows_skip_held_out_tasks_and_budget_stops() -> None:
    run = {"settings": {"env_budget": {"tool_limit": 300}},
           "tasks": [{"task": "a", "split": "train", "n": 8, "passes": 2, "budget_stops": {"student": 3}, "faults": 3, "lang": "go"},
                     {"task": "h", "split": "heldout", "n": 8, "passes": 3, "faults": 2, "lang": "go"}]}
    rows = training_rows([run], {})
    assert [(t, k, n) for t, _, k, n in rows] == [("a", 2, 5)]


# ---- views ---------------------------------------------------------------------------------------------------------------
class FakeRun:
    def __init__(self):
        self.tables, self.lines_, self.summ, self.bars = {}, {}, {}, {}

    def table(self, key, columns, rows, *, step=None):
        self.tables[key] = (list(columns), [list(r) for r in rows])

    def lines(self, key, xs, ys, names, *, title, xname="x", step=None):
        self.lines_[key] = (list(xs), ys, list(names))

    def bar(self, key, labels, values, *, title, label="label", value="value", step=None):
        self.bars[key] = (list(labels), list(values))

    def summary(self, data):
        self.summ.update(data)


def test_views_log_p_hat_band_actions_and_predictor_error() -> None:
    tr = DifficultyTracker(decay=1.0)
    rows = []
    for step in range(3):
        tr.observe("a", step, [Draw(True, 1.0), Draw(False, 0.5)])
        rows += tr.rows(step)
    run = FakeRun()
    views.p_hat_over_time(run, rows)
    views.actions(run, [control("a", _rate(4, 8)).to_json()])
    views.predictor(run, {"mae": 0.1, "mae_baseline": 0.2, "tasks": 2, "predictions": [{"task": "a", "p_true": 0.5, "p_pred": 0.4, "baseline": 0.3, "n": 8}]})
    assert "difficulty/p_hat" in run.tables and run.lines_["difficulty/p_hat_lines"][2] == ["a"] and run.summ["difficulty/in_band"] == 1
    assert run.tables["difficulty/actions"][1][0][1] == "keep" and run.summ["difficulty/predictor_mae"] == 0.1
