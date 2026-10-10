"""M12 trainer math: every mismatch correction against the article's worked numbers (research m11-m12/06 §5, §8
items 31–34), loss aggregation (DAPO's length example), group advantages, critic GAE (skip-observation, cross-segment,
truncation bootstrap), DAPO overlong shaping."""

from __future__ import annotations

import math

import pytest
import torch

from crucible.train.mismatch import Correction, binary_kl, correct
from crucible.train.objectives import aggregate, gae, group_advantages, length_adaptive_lambda, overlong_penalty


def lp(p):
    return torch.tensor([[math.log(x) for x in p]], dtype=torch.float64)


def test_kpop_binary_kl_golden():
    a = binary_kl(torch.tensor(5e-5, dtype=torch.float64), torch.tensor(1e-5, dtype=torch.float64))
    b = binary_kl(torch.tensor(0.9, dtype=torch.float64), torch.tensor(0.45, dtype=torch.float64))
    assert abs(float(a) - 4.05e-5) < 1e-7 and abs(float(b) - 0.4534) < 1e-4
    old, roll = lp([5e-5, 0.9]), lp([1e-5, 0.45])
    logp = old.clone().requires_grad_(True)
    t = correct(Correction("kpop", {"phi": 0.01}), logp=logp, rollout_logp=roll, old_logp=old, adv=torch.ones(1, 2, dtype=torch.float64), mask=torch.ones(1, 2))
    assert t.kept.tolist() == [[True, False]]  # ratio 5 kept (tiny probabilities), ratio 2 masked (large probabilities)


def test_icepop_gate_and_calibration():
    old, roll = lp([0.6, 0.07]), lp([0.1, 0.1])  # r_dis = 6 (masked), 0.7 (calibrated)
    logp = old.clone().requires_grad_(True)
    t = correct(Correction("icepop"), logp=logp, rollout_logp=roll, old_logp=old, adv=torch.ones(1, 2, dtype=torch.float64), mask=torch.ones(1, 2))
    assert float(t.objective[0, 0]) == 0.0 and abs(float(t.objective[0, 1]) - 0.7) < 1e-9  # r_stale = 1 here


def test_dis_masks_both_sides_regardless_of_sign_and_needs_no_old():
    roll = lp([0.5, 0.5, 0.5, 0.5])
    logp = lp([0.05, 0.5, 1.0, 0.999]).requires_grad_(True)  # r = 0.1 (below 0.2), 1, 2, ~2
    adv = torch.tensor([[1.0, -1.0, 1.0, -1.0]], dtype=torch.float64)
    t = correct(Correction("dis", {"eps_low": 0.8, "eps_high": 3.0}), logp=logp, rollout_logp=roll, adv=adv, mask=torch.ones(1, 4))
    assert t.kept.tolist() == [[False, True, True, True]]
    t.objective.sum().backward()
    assert logp.grad[0, 0] == 0 and logp.grad[0, 1] == -1.0  # gradient = sg(r) · Â on kept tokens


def test_mispo_trajectory_filter_golden():
    def run(g):
        roll = torch.zeros(1, 4, dtype=torch.float64)
        old = torch.full((1, 4), math.log(g), dtype=torch.float64)  # geometric mean of x_t = g
        return correct(Correction("mispo"), logp=old.clone().requires_grad_(True), rollout_logp=roll, old_logp=old,
                       adv=torch.ones(1, 4, dtype=torch.float64), mask=torch.ones(1, 4))
    assert not run(0.995).kept.any() and run(0.999).kept.all()
    roll = torch.zeros(1, 3, dtype=torch.float64)
    old = torch.tensor([[math.log(2.1), -0.001, -0.001]], dtype=torch.float64)
    t = correct(Correction("mispo", {"traj_low": 0.0, "traj_high": 10.0}), logp=old.clone().requires_grad_(True), rollout_logp=roll, old_logp=old,
                adv=torch.ones(1, 3, dtype=torch.float64), mask=torch.ones(1, 3))
    assert t.kept.tolist() == [[False, True, True]]  # token ratio 2.1 masked


def test_hispo_bounds_negative_advantage_blowup():
    roll = lp([0.01])
    logp = lp([0.5]).requires_grad_(True)  # r = 50
    t = correct(Correction("hispo", {"delta_seq": 1e9, "delta_tok": 1e9, "eps_neg_high": 0.28}), logp=logp, rollout_logp=roll, old_logp=roll,
                adv=torch.tensor([[-1.0]], dtype=torch.float64), mask=torch.ones(1, 1))
    assert abs(float(t.objective[0, 0]) - (-1.28)) < 1e-9  # dual clip: bounded at (1 + ε_neg_high)·Â


def test_cispo_stop_gradient():
    old = lp([0.4])
    logp = lp([0.6]).requires_grad_(True)  # r = 1.5 -> clipped to 1.28
    t = correct(Correction("cispo"), logp=logp, rollout_logp=old, old_logp=old, adv=torch.tensor([[2.0]], dtype=torch.float64), mask=torch.ones(1, 1))
    t.objective.sum().backward()
    assert abs(float(logp.grad[0, 0]) - 1.28 * 2.0) < 1e-9  # ∇ = r̂ · Â · ∇log π (no (1 + log π) factor)


def test_tis_and_ppo_and_gspo_shapes():
    old, roll = lp([0.5, 0.5]), lp([0.1, 0.5])
    logp = old.clone().requires_grad_(True)
    t = correct(Correction("tis", {"C": 2.0}), logp=logp, rollout_logp=roll, old_logp=old, adv=torch.ones(1, 2, dtype=torch.float64), mask=torch.ones(1, 2))
    assert abs(float(t.objective[0, 0]) - 2.0) < 1e-9 and abs(float(t.objective[0, 1]) - 1.0) < 1e-9  # r_dis 5 truncated at C
    g = correct(Correction("gspo"), logp=old.clone().requires_grad_(True), rollout_logp=roll, old_logp=old, adv=torch.ones(1, 2, dtype=torch.float64),
                mask=torch.ones(1, 2))
    assert torch.allclose(g.objective, torch.ones(1, 2, dtype=torch.float64))


def test_needs_old_refuses_without_source():
    roll = lp([0.5])
    with pytest.raises(ValueError):
        correct(Correction("ppo"), logp=roll.clone().requires_grad_(True), rollout_logp=roll, adv=torch.ones(1, 1), mask=torch.ones(1, 1))
    correct(Correction("ppo", old_logprob_source="rollout"), logp=roll.clone().requires_grad_(True), rollout_logp=roll, adv=torch.ones(1, 1), mask=torch.ones(1, 1))


def test_rollout_probs_diff_logged_under_none_like_ppo():
    old, roll = lp([0.5, 0.2]), lp([0.4, 0.2])
    t = correct(Correction("ppo"), logp=old.clone().requires_grad_(True), rollout_logp=roll, old_logp=old, adv=torch.ones(1, 2, dtype=torch.float64),
                mask=torch.ones(1, 2))
    assert abs(t.metrics["rollout_probs_diff"] - 0.05) < 1e-9 and "r_dis/p50" in t.metrics


def test_dapo_length_example():
    J = torch.zeros(2, 10_000, dtype=torch.float64, requires_grad=True)
    mask = torch.zeros(2, 10_000)
    mask[0, :100] = 1
    mask[1, :] = 1
    aggregate(J, mask, "seq_mean_token_mean").backward()
    assert abs(float(J.grad[0, 0]) - 5e-3) < 1e-12 and abs(float(J.grad[1, 0]) - 5e-5) < 1e-12
    J.grad = None
    aggregate(J, mask, "token_mean_global").backward()
    assert abs(float(J.grad[0, 0]) - 1 / 10_100) < 1e-12 and abs(float(J.grad[1, 0]) - 1 / 10_100) < 1e-12  # 9.9e-5 each


def test_group_advantages():
    a = group_advantages([1.0, 0.0, None, 1.0], ["g", "g", "g", "h"])
    assert a == [0.5, -0.5, 0.0, 0.0]  # masked excluded, singleton group 0
    r = group_advantages([1.0, 0.0, 0.0], ["g"] * 3, estimator="rloo")
    assert abs(r[0] - 1.0) < 1e-12 and abs(r[1] + 0.5) < 1e-12


def test_gae_standard_skip_observation_and_segments():
    # standard: 3 trainable tokens, reward 1 at the end, V = 0, γ = λ = 1 -> all advantages 1
    a, _ = gae(torch.tensor([0, 0, 1.0]), torch.zeros(3), torch.tensor([True, True, True]))
    assert a.tolist() == [1.0, 1.0, 1.0]
    # skip-observation: inserting observation tokens with arbitrary values changes nothing
    vals = torch.tensor([0.2, 9.0, -9.0, 0.4, 0.0])
    a2, _ = gae(torch.tensor([0, 0, 0, 0, 1.0]), vals, torch.tensor([True, False, False, True, True]), gamma=1.0, lam=0.9)
    a3, _ = gae(torch.tensor([0, 0, 1.0]), torch.tensor([0.2, 0.4, 0.0]), torch.tensor([True, True, True]), gamma=1.0, lam=0.9)
    assert torch.allclose(a2[[0, 3, 4]], a3)
    # cross-segment: two segments of 2 tokens, shared terminal R = 1, γλ = 0.5 -> segment 0 scaled by 0.5^2
    seg = torch.tensor([0, 0, 1, 1])
    a4, _ = gae(torch.zeros(4), torch.zeros(4), torch.ones(4, dtype=torch.bool), gamma=1.0, lam=0.5, segments=seg, shared_terminal=1.0)
    loc = [0.5, 1.0]  # local GAE with λ = 0.5 inside a segment
    assert torch.allclose(a4, torch.tensor([loc[0] * 0.25, loc[1] * 0.25, loc[0], loc[1]]))
    # one segment == standard GAE
    a5, _ = gae(torch.zeros(3), torch.zeros(3), torch.ones(3, dtype=torch.bool), lam=0.5, segments=torch.zeros(3, dtype=torch.long), shared_terminal=1.0)
    a6, _ = gae(torch.tensor([0, 0, 1.0]), torch.zeros(3), torch.ones(3, dtype=torch.bool), lam=0.5)
    assert torch.allclose(a5, a6)
    # truncation bootstrap: V(s_T) replaces 0
    a7, _ = gae(torch.zeros(2), torch.zeros(2), torch.ones(2, dtype=torch.bool), truncated=True, bootstrap_value=0.7)
    assert torch.allclose(a7, torch.tensor([0.7, 0.7]))


def test_overlong_and_lambda():
    assert overlong_penalty(16_000) == 0.0 and overlong_penalty(20_480) == -1.0
    assert overlong_penalty(18_432) == -0.5 and overlong_penalty(30_000) == -1.0
    assert abs(length_adaptive_lambda(100) - (1 - 1 / 150)) < 1e-12


def test_dp_balance_and_micro_batches():
    from crucible.train.batching import balance_partitions, imbalance, loss_scale, micro_batches

    lengths = [8000, 7000, 300, 250, 6000, 400, 5000, 120, 90, 4000, 3500, 200, 150, 3000, 2900, 100]
    parts = balance_partitions(lengths, 4)
    assert sorted(i for p in parts for i in p) == list(range(len(lengths)))
    assert imbalance(lengths, parts) <= 1.1  # checklist item 19
    mbs = micro_batches(lengths, parts[0], token_budget=8192)
    assert all(m.tokens <= 8192 or len(m.indices) == 1 for m in mbs)
    total = sum(lengths)
    assert abs(sum(loss_scale(m.tokens, total) for p in parts for m in micro_batches(lengths, p, 8192)) - 1.0) < 1e-12


def test_token_credit_masks_frozen_and_masked():
    from crucible.train.credit import token_credit
    from crucible.train.rollout import RolloutRecord, Turn

    ro = RolloutRecord(rollout_id="r", sample="s", task="t", harness="h", policy="p", reward=1.0,
                       turns=[Turn(call_id="a:c1", actor="agent", completion_token_ids=[1, 2], logprobs=[-0.1, -0.2], policy_version=3, advantage=0.25),
                              Turn(call_id="b:c1", actor="sub", completion_token_ids=[3], logprobs=[-0.3], policy_version=3)])
    tc = token_credit(ro, 0.5, turn_weight=1.0, frozen_actors=["sub"])
    assert tc[0].advantages == [0.75, 0.75] and tc[0].loss_mask == [1, 1] and tc[1].loss_mask == [0]
    ro.mask_sample = True
    assert all(m == 0 for x in token_credit(ro, 0.5) for m in x.loss_mask)


def test_version_spans_validation():
    from crucible.train.rollout import RolloutRecord, Turn

    base = {"call_id": "c", "actor": "a", "prompt_token_ids": [1], "completion_token_ids": [5, 6, 7, 8], "logprobs": [-1.0] * 4, "policy_version": 3}
    ok = Turn(**base, version_spans=[(0, 3), (2, 4)])
    bad = Turn(**base, version_spans=[(1, 3)])  # does not start at token 0
    assert ok.spans_valid() and ok.versions() == [3, 4] and not bad.spans_valid()
    ro = RolloutRecord(rollout_id="r", sample="s", task="t", harness="h", policy="p", reward=1.0, turns=[ok])
    assert ro.valid_for_training()[0] and ro.oldest_version() == 3
    ro.turns = [bad]
    assert not ro.valid_for_training()[0]


def test_micro_batch_budget_counts_forward_tokens_not_just_completions():
    """A multi-turn call's forward pass is prompt + completion: two 35-token prompts with 2-token completions do not fit
    one 40-token micro batch (placement by completion length alone said they did: 4 tokens)."""
    import torch

    from crucible.train.engine import TinyPolicy
    from crucible.train.rollout import RolloutRecord, Turn
    from crucible.train.trainer import TorchTrainer, TrainerConfig

    recs = [RolloutRecord(rollout_id=f"r{i}", sample=f"r{i}", task="t", harness="h", policy="p", reward=float(i),
                          turns=[Turn(call_id=f"r{i}:c", actor="agent", prompt_token_ids=list(range(35)), completion_token_ids=[1, 2], logprobs=[-1.0, -1.0],
                                      policy_version=0)]) for i in range(2)]
    tr = TorchTrainer(TinyPolicy(48, 8, seed=0), TrainerConfig(token_budget=40))
    m = tr.step(recs, ["g", "g"])
    assert m["train/micro_batches"] == 2 and m["train/tokens"] == 4 and m["train/forward_tokens"] == 74
    assert torch.isfinite(torch.tensor(m["train/loss"]))


def test_trainer_refuses_implicit_old_logprobs_for_multi_version_rollouts():
    """Checklist #34: a π_θold correction with no source recomputes only for single-version rollouts; a rollout that
    spans versions needs an explicit old_logprob_source."""
    import pytest

    from crucible.train.engine import TinyPolicy
    from crucible.train.mismatch import Correction
    from crucible.train.rollout import RolloutRecord, Turn
    from crucible.train.trainer import TorchTrainer, TrainerConfig

    def rec(i, versions):
        turns = [Turn(call_id=f"r{i}:c{k}", actor="agent", prompt_token_ids=[1, 2, 3] if k == 0 else [4], prompt_is_delta=k > 0, completion_token_ids=[5, 6],
                      logprobs=[-1.0, -1.0], policy_version=v) for k, v in enumerate(versions)]
        return RolloutRecord(rollout_id=f"r{i}", sample=f"r{i}", task="t", harness="h", policy="p", reward=float(i % 2), turns=turns)

    multi = [rec(0, [0, 1]), rec(1, [1, 1])]
    with pytest.raises(ValueError, match="span several policy versions"):
        TorchTrainer(TinyPolicy(16, 8), TrainerConfig(correction=Correction("ppo"))).step(multi, ["g", "g"])
    m = TorchTrainer(TinyPolicy(16, 8), TrainerConfig(correction=Correction("ppo", old_logprob_source="recompute_current"))).step(multi, ["g", "g"])
    assert m["train/items"] == 4
    m = TorchTrainer(TinyPolicy(16, 8), TrainerConfig(correction=Correction("ppo"))).step([rec(0, [1, 1]), rec(1, [1, 1])], ["g", "g"])
    assert m["train/items"] == 4  # single-version rollouts: the implicit recompute is valid


def test_top_p_candidates_renormalize_training_logprobs_to_the_engine():
    """Checklist #37: with top_p < 1 the engine's behaviour logprobs are of the renormalized nucleus; the trainer's
    logprobs renormalized over the recorded candidates match them within 1e-5, and unrenormalized ones do not."""
    import asyncio

    import torch

    from crucible.train.engine import GenRequest, TinyPolicy, TorchEngine
    from crucible.train.rollout import RolloutRecord, Turn
    from crucible.train.trainer import TorchTrainer, TrainerConfig

    model = TinyPolicy(32, 16, seed=3)
    eng = TorchEngine(model, capacity=2, numerics="fp32")
    res = asyncio.run(eng.generate(GenRequest(rid="r", call_id="c", prompt_ids=[1, 5, 2], max_tokens=12, temperature=0.9, top_p=0.6, seed=7)))
    assert res.candidates and len(res.candidates) == len(res.completion_ids) and all(t in c for t, c in zip(res.completion_ids, res.candidates, strict=True))
    assert any(len(c) < 32 for c in res.candidates)  # the nucleus actually truncated
    with torch.no_grad():
        renorm = model.token_logprobs([1, 5, 2], res.completion_ids, 0.9, candidates=res.candidates)
        plain = model.token_logprobs([1, 5, 2], res.completion_ids, 0.9)
    eng_lp = torch.tensor(res.logprobs, dtype=renorm.dtype)
    assert float((renorm - eng_lp).abs().max()) < 1e-5
    assert float((plain - eng_lp).abs().max()) > 1e-3  # the bias renormalization removes

    turn = {"call_id": "c", "actor": "agent", "prompt_token_ids": [1, 5, 2], "completion_token_ids": res.completion_ids, "logprobs": res.logprobs,
            "policy_version": 0, "sampling": {"temperature": 0.9, "top_p": 0.6}}
    ok = RolloutRecord(rollout_id="a", sample="a", task="t", harness="h", policy="p", reward=1.0, turns=[Turn(**turn, candidates=res.candidates)])
    bad = RolloutRecord(rollout_id="b", sample="b", task="t", harness="h", policy="p", reward=0.0, turns=[Turn(**turn)])
    assert ok.valid_for_training()[0] and not bad.valid_for_training()[0]
    tr = TorchTrainer(model, TrainerConfig(temperature=0.9))
    assert tr.step([ok, ok.model_copy(update={"rollout_id": "a2", "reward": 0.0})], ["g", "g"])["train/items"] == 2
    import pytest

    with pytest.raises(ValueError, match="no complete candidate sets"):
        tr.step([bad, ok], ["g", "g"])


def test_r3_routing_replay_removes_the_moe_routing_mismatch():
    """Checklist #36: a full-bf16 engine flips some router choices of a tiny MoE policy; without replay the trainer evaluates
    a different expert path on those tokens; with routing_replay='r3' it replays the engine's experts and the logprob gap
    shrinks to plain numerics. R3 refuses records missing routing; routing confidence is logged."""
    import asyncio

    import pytest
    import torch

    from crucible.train.engine import GenRequest, TinyMoEPolicy, TorchEngine
    from crucible.train.rollout import RolloutRecord, Turn
    from crucible.train.trainer import TorchTrainer, TrainerConfig

    model = TinyMoEPolicy(32, 16, seed=2, experts=4, k=1, router_scale=0.02)
    eng = TorchEngine(model, capacity=4, numerics="bf16_full")  # bf16 weights and activations: router choices flip
    gap_plain, gap_r3, flips, recs = [], [], 0, []
    for i in range(12):
        prompt = [1, 4 + i % 20, 2]
        res = asyncio.run(eng.generate(GenRequest(rid=f"r{i}", call_id=f"c{i}", prompt_ids=prompt, max_tokens=10, temperature=1.0, seed=i)))
        assert res.routing and len(res.routing) == len(res.completion_ids)
        with torch.no_grad():
            plain = model.token_logprobs(prompt, res.completion_ids)
            own = [model.last_routes[len(prompt) - 1 + t] for t in range(len(res.completion_ids))]
            r3 = model.token_logprobs(prompt, res.completion_ids, routing=res.routing)
        eng_lp = torch.tensor(res.logprobs, dtype=plain.dtype)
        for t, (a, b) in enumerate(zip(own, res.routing, strict=True)):
            if a != b:
                flips += 1
                gap_plain.append(abs(float(plain[t] - eng_lp[t])))
                gap_r3.append(abs(float(r3[t] - eng_lp[t])))
        recs.append(RolloutRecord(rollout_id=f"r{i}", sample=f"r{i}", task="t", harness="h", policy="p", reward=float(i % 2),
                                  turns=[Turn(call_id=f"c{i}", actor="agent", prompt_token_ids=prompt, completion_token_ids=res.completion_ids, logprobs=res.logprobs,
                                              policy_version=0, routing=res.routing)]))
    assert flips >= 3, flips  # the engine's numerics really changed expert choices
    assert sum(gap_r3) / len(gap_r3) < 0.5 * sum(gap_plain) / len(gap_plain), (sum(gap_r3), sum(gap_plain))
    m = TorchTrainer(model, TrainerConfig(routing_replay="r3")).step(recs, ["g"] * len(recs))
    assert m["train/items"] == len(recs) and 0.25 <= m["moe/routing_confidence"] <= 1.0
    stripped = [r.model_copy(update={"turns": [t.model_copy(update={"routing": None}) for t in r.turns]}) for r in recs]
    with pytest.raises(ValueError, match="needs Turn.routing for 100%"):
        TorchTrainer(model, TrainerConfig(routing_replay="r3")).step(stripped, ["g"] * len(recs))
    assert TorchTrainer(model, TrainerConfig()).step(stripped, ["g"] * len(recs))["train/items"] == len(recs)  # without R3 nothing is required
