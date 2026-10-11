"""M6 credit: validity statistics, C1 potential and shaping, the gated composition, per-turn advantages,
steps.jsonl, the C6 mask and the R1-R6 properties."""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from crucible.core.schema.milestone import Milestone, MilestoneCheck
from crucible.core.schema.reward import InfraStatus
from crucible.core.schema.state import Query
from crucible.credit import steps as steps_mod
from crucible.credit.compose import Coefficients, Episode, compose, scalar, turn_advantages
from crucible.credit.operators.infra_mask import mask_for
from crucible.credit.properties import check_all, r1_oracle, r2_noop, r3_negatives, r4_oscillation, r5_stubbing, r6_state_only
from crucible.credit.validity import auc, paired_bootstrap, time_split, validate
from crucible.workflow.potential import potential, shaping, trajectory
from crucible.workflow.steps import Step


def ms(*spec):
    return [Milestone(id=i, requires=list(req), weight=w, check=MilestoneCheck(app="a", query=Query(op="x"))) for i, req, w in spec]


BUILD_TEST = ms(("build", (), 1.0), ("test", ("build",), 2.0))


def test_auc_matches_reference_values():
    assert auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75  # sklearn's documented example
    assert auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5 and auc([0.5, 0.5, 0.9], [0, 1, 1]) == 0.75 and auc([1, 2], [1, 1]) is None


def test_time_split_and_validity_rule():
    tr, ho = time_split(list(range(10)), time=lambda x: x, holdout=0.3)
    assert tr == list(range(7)) and ho == [7, 8, 9]
    keep = validate("m", [0.1, 0.2, 0.8, 0.9, 0.3, 0.7], [0, 0, 1, 1, 0, 1])
    assert keep.kept and keep.weight == pytest.approx(1.0)
    weak = validate("w", [0.5, 0.4, 0.5, 0.6], [0, 1, 0, 1])
    assert not weak.kept and "AUC" in weak.reason
    # a direction that flips on one subset is dropped even if the pooled AUC is high
    s = [0.9, 0.8, 0.1, 0.2] * 3 + [0.1, 0.2, 0.9, 0.8]  # predictive on repo r1, reversed on repo r2
    y = [1, 1, 0, 0] * 4
    flip = validate("f", s, y, subsets=["r1"] * 12 + ["r2"] * 4, min_subset=2, min_auc=0.6)
    assert flip.auc and flip.auc >= 0.6 and not flip.kept and "flips" in flip.reason, flip
    pb = paired_bootstrap([0.1, 0.2, 0.8, 0.9] * 5, [0.5, 0.4, 0.6, 0.5] * 5, [0, 0, 1, 1] * 5, groups=[i // 4 for i in range(20)], n=300)
    assert pb["diff"] == pytest.approx(0.125) and pb["lo"] >= 0 and pb["p_not_better"] < 0.5  # AUC 1.0 vs 0.875


def test_potential_prerequisites_regression_and_telescoping():
    assert potential(BUILD_TEST, {"build": 0, "test": 1})[0] == 0.0  # test before build: stubbing earns nothing
    assert potential(BUILD_TEST, {"build": 1, "test": None}) == (pytest.approx(1 / 3), True)
    tr = trajectory(BUILD_TEST, [{"build": 0, "test": 0}, {"build": 1, "test": 0}, {"build": 1, "test": 1}, {"build": 0, "test": 1}])
    assert [round(s.phi, 3) for s in tr] == [0.0, 0.333, 1.0, 0.0] and tr[3].lost == ["build", "test"]
    assert sum(shaping([s.phi for s in tr], initial=0.2)) == pytest.approx(-0.2)  # telescopes to -Φ(s0)


def test_compose_bounds_gates_and_mask():
    ok = compose(gates={"build": True}, outcome=True, phi=1.0, rubric=1.0, efficiency=1.0)
    fail = compose(gates={"build": True}, outcome=False, phi=1.0)
    assert ok.reward >= 1.0 and fail.reward <= Coefficients().kappa and fail.reward < ok.reward
    assert compose(gates={"build": False}, outcome=True, phi=1.0).reward == 0.0
    masked = compose(gates={}, outcome=False, phi=0.4, masked=InfraStatus(status="masked", failure_kind="transport_timeout"))
    assert scalar(masked) is None
    with pytest.raises(ValueError):
        Coefficients(kappa=0.5)
    with pytest.raises(ValueError):
        compose(gates={}, outcome=False, phi=1.4)


def test_turn_advantages_zero_mass_and_masking():
    g = [Episode(reward=1.0, phis=[0.2, 0.6, 1.0]), Episode(reward=0.06, phis=[0.2, 0.2, 0.3]), Episode(reward=None, phis=[0.1, 0.2])]
    for mode in ("delta", "baseline"):
        adv = turn_advantages(g, eta=0.1, mode=mode)
        mu = statistics.fmean([1.0, 0.06])
        assert sum(x - (g[i].reward - mu) for i in range(2) for x in adv[i]) == pytest.approx(0.0, abs=1e-9)
        assert adv[2] == [0.0, 0.0]
    delta = turn_advantages(g, eta=0.1)
    assert delta[0][1] > delta[0][0]  # the turn that moved Φ most gets the most local credit
    assert min(delta[0]) > max(delta[1])  # local credit never reverses the outcome ordering


def test_steps_jsonl_round_trip(tmp_path: Path):
    st = [Step(index=i, seqs=[i], kind=k) for i, k in enumerate(["investigate", "edit", "test"])]
    pot = trajectory(BUILD_TEST, [{"build": 0, "test": 0}, {"build": 1, "test": 0}, {"build": 1, "test": 1}])
    rows = steps_mod.rows_for(st, pot, call_ids=["c1", "c2", "c3"], harness="bash_loop@1")
    steps_mod.write(rows, tmp_path / "steps.jsonl")
    back = steps_mod.read(tmp_path / "steps.jsonl")
    assert [r.call_id for r in back] == ["c1", "c2", "c3"] and steps_mod.to_turn_shaping(back) == pytest.approx([0.0, 1 / 3, 2 / 3])


def test_infra_mask_rules():
    assert mask_for().status == "ok"
    assert mask_for(failures=[{"kind": "transport_timeout"}]).status == "masked"
    assert mask_for(failures=[{"kind": "agent_timeout"}]).status == "ok"  # the policy's own outcome is scored
    assert mask_for(gates={"build": None}).failure_kind == "verifier_error"
    assert mask_for(terminal_phis={"a": None, "b": None}).status == "masked" and mask_for(terminal_phis={"a": None, "b": 1.0}).status == "ok"
    assert mask_for(failures=[{"kind": "crucible:answer_exposed", "mask": True}]).status == "masked"


def test_properties_on_a_toy_reward():
    def reward_of(state):
        phis = {"build": float("built" in state), "test": float("tested" in state)}
        return compose(gates={"compiles": "broken" not in state}, outcome=phis["build"] == phis["test"] == 1.0, phi=potential(BUILD_TEST, phis)[0])

    results = [r1_oracle(reward_of, {"built", "tested"}), r2_noop(reward_of, set()), r3_negatives(reward_of, {"broken": {"built", "tested", "broken"}, "stub": {"tested"}}),
               r4_oscillation(BUILD_TEST, [{"build": 1, "test": 0}, {"build": 1, "test": 1}], [{"build": 1, "test": 0}, {"build": 0, "test": 0}, {"build": 1, "test": 1}]),
               r5_stubbing(BUILD_TEST, {"build": 0, "test": 1}),
               r6_state_only(lambda s, msgs: reward_of(s), {"built"}, [["I fixed everything!"], ["done"]])]
    rep = check_all(results)
    assert rep["ok"], rep
    # a reward that pays for claims fails R6; one that pays the no-op fails R2
    assert not r6_state_only(lambda s, msgs: compose(gates={}, outcome=False, phi=0.5 if "fixed" in msgs[0] else 0.0), set(), [["fixed"], ["x"]]).ok
    assert not r2_noop(lambda s: compose(gates={}, outcome=False, phi=0.5), set()).ok
