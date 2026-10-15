"""M12 audit (docs/audit/2026-10-12/M12-train.md), the learning math: regression tests named by finding id (H1 = the math
hunt, G1-G5 = the source sweeps, H4 = scale-out). Each fails on the M11 commit (3b522cb)."""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.train.critic import CriticConfig, TinyValue
from crucible.train.engine import TinyPolicy
from crucible.train.mismatch import Correction, correct
from crucible.train.trainer import TorchTrainer, TrainerConfig


def _turn(cid="c0", prompt=(1, 2), comp=(3, 4, 5), lp=-1.0, **kw):
    return Turn(sampling=kw.pop("sampling", {"temperature": 1.0}), call_id=cid, actor=kw.pop("actor", "agent"), prompt_token_ids=list(prompt), completion_token_ids=list(comp),
                logprobs=[lp] * len(comp) if not isinstance(lp, list) else lp, policy_version=0, **kw)


def _rec(rid, reward, turns=None, **kw):
    return RolloutRecord(rollout_id=rid, sample=rid, task="t", harness="h", policy="p", reward=reward, turns=turns or [_turn()], **kw)


def _grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


# ---- numerics (H1 V2, G2/G3/G4 P0) ------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["dis", "mimo4", "icepop", "kpop", "hispo", "ppo", "tis", "cispo", "gspo", "mispo"])
def test_H1_V2_extreme_behaviour_logprob_never_makes_nan(name):
    torch.manual_seed(0)
    logp = (torch.randn(2, 6, dtype=torch.float64) * 0.3 - 1).requires_grad_(True)
    roll = logp.detach().clone()
    roll[0, 2] = -9999.0  # an OpenAI-style floor: finite, so the schema accepts it
    roll[1, 4] = -800.0
    t = correct(Correction(name, old_logprob_source="rollout"), logp=logp, rollout_logp=roll, adv=torch.ones(2, 6, dtype=torch.float64),
                mask=torch.ones(2, 6))
    (-t.objective.sum()).backward()
    assert torch.isfinite(logp.grad).all() and torch.isfinite(t.objective).all()


def test_H1_V2_nonfinite_behaviour_logprob_drops_the_token():
    logp = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    roll = torch.tensor([[0.0, float("nan"), -float("inf")]], dtype=torch.float64)
    t = correct(Correction("dis"), logp=logp, rollout_logp=roll, adv=torch.ones(1, 3, dtype=torch.float64), mask=torch.ones(1, 3))
    (-t.objective.sum()).backward()
    assert torch.isfinite(logp.grad).all() and t.metrics["nonfinite_tokens"] == 2 and t.kept.tolist() == [[True, False, False]]


def test_G3_D7_nonfinite_grad_norm_skips_the_step():
    model = TinyPolicy(32, 8, seed=1)
    with torch.no_grad():
        model.head.bias[3] = float("inf")  # a poisoned forward
    before = {k: v.clone() for k, v in model.state_dict().items()}
    tr = TorchTrainer(model, TrainerConfig(lr=0.1))
    m = tr.step([_rec("a", 1.0), _rec("b", 0.0, [_turn(comp=(6, 7))])], ["g", "g"])
    assert m["train/skipped_nonfinite"] == 1.0 and tr.updates == 0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


# ---- validity and temperature gates (G2 H9, G4 #7, H3 V21, H1 V3) ----------------------------------------------------

def test_G2_H9_call_without_logprobs_is_masked_not_trained_at_zero():
    bad = _rec("b", 0.0, [_turn(comp=(6, 7))])
    bad.turns[0].logprobs = None
    tr = TorchTrainer(TinyPolicy(32, 8, seed=1), TrainerConfig())
    m = tr.step([_rec("a", 1.0), bad, _rec("c", 0.0, [_turn(comp=(8,))])], ["g", "g", "g"])
    assert m["train/records_invalid"] == 1.0 and m["train/tokens"] == 4.0  # a's 3 + c's 1; b's 2 never trained


def test_H1_V3_trainer_refuses_a_different_sampling_temperature():
    tr = TorchTrainer(TinyPolicy(32, 8), TrainerConfig(temperature=1.0))
    with pytest.raises(ValueError, match="temperature 0.6"):
        tr.step([_rec("a", 1.0, [_turn(sampling={"temperature": 0.6})]), _rec("b", 0.0)], ["g", "g"])


# ---- aggregation over rollouts, micro-batched (H1 V4, V13, G2 H4, H8, G1 #3, G4 #7, G5 D11) ------------------------------

def _multi_call_batch(top_p=1.0):
    out = []
    for i, (r, n) in enumerate(((1.0, 2), (0.0, 1), (0.5, 3), (0.0, 2))):
        turns = []
        for k in range(n):
            comp = [3 + (i + k) % 20, 4, 5][: 1 + (i + k) % 3]
            cand = [[c, (c + 1) % 32] for c in comp] if top_p < 1 else None
            turns.append(_turn(f"c{k}", prompt=[1, 2] if k == 0 else [9], comp=comp, prompt_is_delta=k > 0, sampling={"temperature": 1.0, "top_p": top_p},
                               candidates=cand))
        out.append(_rec(f"r{i}", r, turns))
    return out, ["g", "g", "h", "h"]


@pytest.mark.parametrize("agg", ["seq_mean_token_mean", "prompt_mean", "const_max_len", "token_mean_global"])
def test_H1_V4_aggregation_unit_is_the_rollout_whatever_the_rows(agg):
    """The same rollouts as segment rows (top_p = 1) and as per-call rows (top_p < 1 with full candidate sets that
    renormalize to nothing new for the gradient's unit): the per-rollout weights must not depend on the row split."""
    from crucible.train.objectives import row_weights

    seg = row_weights([5, 3], ["r0", "r1"], [0, 0], agg)
    call = row_weights([2, 3, 3], ["r0", "r0", "r1"], [0, 0, 0], agg)
    assert sum(seg[:1]) * 5 == pytest.approx(call[0] * 2 + call[1] * 3)  # rollout r0's total weight is unchanged
    recs, groups = _multi_call_batch()
    a = TorchTrainer(TinyPolicy(32, 8, seed=2), TrainerConfig(loss_agg=agg, item_unit="segment", lr=0.0))
    b = TorchTrainer(TinyPolicy(32, 8, seed=2), TrainerConfig(loss_agg=agg, item_unit="call", lr=0.0))
    la, lb = a.step(recs, groups)["train/loss"], b.step(recs, groups)["train/loss"]
    assert la == pytest.approx(lb, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("agg", ["seq_mean_token_mean", "prompt_mean", "const_max_len"])
def test_H1_V13_every_aggregation_micro_batches_to_the_full_batch_gradient(agg):
    recs, groups = _multi_call_batch()
    grads = []
    for budget in (10_000, 3):
        model = TinyPolicy(32, 8, seed=4)
        tr = TorchTrainer(model, TrainerConfig(loss_agg=agg, token_budget=budget, lr=0.0, grad_clip=1e9))
        m = tr.step(recs, groups)
        grads.append(_grads(model))
        if budget == 3:
            assert m["train/micro_batches"] > 2  # really split
    assert torch.allclose(grads[0], grads[1], atol=1e-7)


def test_G1_3_gspo_defaults_to_its_sequence_unit_and_refuses_token_mean():
    from crucible.train.objectives import resolve_agg

    assert resolve_agg("auto", "gspo") == "seq_mean_token_mean" and resolve_agg("auto", "hispo") == "const_max_len"
    with pytest.raises(ValueError, match="gspo"):
        resolve_agg("token_mean_global", "gspo")
    # GSPO's objective value: mean over rollouts of min(s·A, clip(s)·A), the sequence ratio over the rollout's tokens
    logp = torch.tensor([[-1.0, -1.2, 0.0], [-0.5, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([[-1.1, -1.25, 0.0], [-0.5, 0.0, 0.0]], dtype=torch.float64)
    mask = torch.tensor([[1.0, 1, 0], [1, 0, 0]])
    adv = torch.tensor([[1.0, 1, 0], [-1, 0, 0]], dtype=torch.float64)
    t = correct(Correction("gspo", {"eps_low": 1.0, "eps_high": 1.0}), logp=logp, rollout_logp=old, old_logp=old, adv=adv, mask=mask,
                seq_ids=torch.tensor([0, 1]))
    s0 = math.exp(((-1.0 + 1.1) + (-1.2 + 1.25)) / 2)
    assert float((t.objective[0].sum() / 2 + t.objective[1].sum()) / 2) == pytest.approx((s0 * 1 + 1.0 * -1) / 2)


def test_G1_3_hispo_const_max_len_is_per_group():
    from crucible.train.objectives import row_weights

    # group g: rollouts of 2 and 4 tokens -> 1/(2·4) each; group h: one rollout of 1 token -> 1/(1·1); two groups -> /2
    w = row_weights([2, 4, 1], ["a", "b", "c"], ["g", "g", "h"], "const_max_len")
    assert w == pytest.approx([1 / 16, 1 / 16, 1 / 2])
    assert row_weights([2, 4, 1], ["a", "b", "c"], ["g", "g", "h"], "const_max_len", max_len=10) == pytest.approx([1 / 40, 1 / 40, 1 / 20])


def test_G2_H4_sequence_gates_span_the_rollout_not_the_call():
    """MIS-PO's trajectory gate on a rollout split into two rows: one row drifts, the other compensates; per call it
    would reject one row, per rollout it keeps both."""
    logp = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    old = torch.zeros(2, 2, dtype=torch.float64)
    roll = torch.tensor([[0.004, 0.004], [-0.004, -0.004]], dtype=torch.float64)  # x_t = exp(old - roll)
    c = Correction("mispo")
    per_row = correct(c, logp=logp, rollout_logp=roll, old_logp=old, adv=torch.ones(2, 2, dtype=torch.float64), mask=torch.ones(2, 2))
    per_seq = correct(c, logp=logp, rollout_logp=roll, old_logp=old, adv=torch.ones(2, 2, dtype=torch.float64), mask=torch.ones(2, 2),
                      seq_ids=torch.tensor([0, 0]))
    assert per_row.kept.sum() == 2 and per_seq.kept.sum() == 4


# ---- group stage (H1 V1, V16, G2 H1, G1 #8) --------------------------------------------------------------------------

def test_H1_V1_grs_then_gar_keeps_every_passing_solution_positive():
    from crucible.train.group_ops import bt_pairwise, compose, gar, grs

    def recs():
        return [_rec("best", 1.0), _rec("ok", 1.0), _rec("f1", 0.0), _rec("f2", 0.0)]

    rub = {"best": (1.0, 1.0), "ok": (0.9, 0.95)}
    out = asyncio.run(compose(grs(lambda r: rub.get(r.rollout_id)), gar())("g", recs()))
    adv = {r.rollout_id: r.advantage for r in out}
    assert adv["best"] > adv["ok"] > 0 > adv["f1"] == adv["f2"]
    assert sum(adv.values()) == pytest.approx(0.0, abs=1e-12)
    bt = asyncio.run(compose(bt_pairwise(lambda r, ref: {"best": 0.9, "ok": 0.6}.get(r.rollout_id, 0.2), "ref"), gar())("g", recs()))
    a = {r.rollout_id: r.advantage for r in bt}
    assert a["best"] > a["ok"] > 0 > a["f1"]  # bt→gar used to zero the whole group


def test_H1_V16_group_ops_never_score_masked_members():
    from crucible.train.group_ops import tournament, zero_hacks

    ms = [_rec("a", 1.0), _rec("b", 0.0), _rec("m", None, mask_sample=True)]
    out = asyncio.run(zero_hacks(lambda r: True)("g", ms))
    assert out[2].reward is None
    ms = [_rec("a", 1.0), _rec("b", 0.0), _rec("m", None, mask_sample=True)]
    out = asyncio.run(tournament(lambda x, y: 1.0 if x.rollout_id < y.rollout_id else 0.0)("g", ms))
    assert out[2].reward is None and out[0].reward == 1.0 and out[1].reward == 0.0  # 1 game each, not 2


def test_G1_8_one_gar_operator():
    from crucible.credit.gar import gar as credit_gar
    from crucible.train.group_ops import gar

    q = {"p1": 1.0, "p2": 0.5, "p3": 0.25}
    ms = [_rec("p1", 1.0), _rec("p2", 1.0), _rec("p3", 1.0), _rec("f", 0.0)]
    a = [m.advantage for m in asyncio.run(gar(lambda m: q[m.rollout_id])("g", ms))]
    b, _ = credit_gar([1, 1, 1, 0], [1.0, 0.5, 0.25, None])
    assert a == pytest.approx(b) and sum(a[:3]) == pytest.approx(0.75)


# ---- critic (G1 P0, H1 V11, V12) -------------------------------------------------------------------------------------

def test_G1_P0_truncated_rollout_bootstraps_instead_of_adding_the_reward():
    critic = TinyValue(32, 8, seed=1)
    with torch.no_grad():
        critic.head.bias.fill_(0.5)
    rec = _rec("r", 1.0, [_turn(comp=(6, 7, 8))], truncated=True)
    tr = TorchTrainer(TinyPolicy(32, 8), TrainerConfig(item_unit="call"), critic=critic, critic_cfg=CriticConfig(lam_policy=1.0))
    tr._items([rec], ["g"])
    ret = tr._critic_data[0][2]
    assert float(ret.max()) <= 0.5 + 1e-6  # the critic target is V(s_T) = 0.5, never R + V = 1.5
    keep = TorchTrainer(TinyPolicy(32, 8), TrainerConfig(item_unit="call"), critic=critic, critic_cfg=CriticConfig(lam_policy=1.0, truncation="reward"))
    keep._items([rec], ["g"])
    assert float(keep._critic_data[0][2][-1]) == pytest.approx(1.0)


def test_H1_V11_critic_mode_trains_on_the_shaped_reward():
    critic = TinyValue(32, 8, seed=1)
    fast = _rec("f", 1.0, timings={"active_s": 1.0})
    slow = _rec("s", 1.0, timings={"active_s": 3.0})
    tr = TorchTrainer(TinyPolicy(32, 8), TrainerConfig(item_unit="call", speed_coef=0.5), critic=critic, critic_cfg=CriticConfig(lam_policy=1.0))
    items = tr._items([fast, slow], ["g", "g"])
    assert items[0].adv[-1] > items[1].adv[-1]  # the speed term reaches the critic's reward
    with pytest.raises(ValueError, match="group mode"):
        staged = _rec("x", 1.0, advantage=0.4)
        tr._items([staged], ["g"])


def test_H1_V11_reward_to_go_refuses_to_overwrite_group_stage_advantages():
    tr = TorchTrainer(TinyPolicy(32, 8), TrainerConfig(credit_unit="reward_to_go"))
    with pytest.raises(ValueError, match="group stage"):
        tr._items([_rec("a", 1.0, advantage=0.9), _rec("b", 0.0, advantage=-0.9)], ["g", "g"])


def test_H1_V9_speed_bonus_needs_a_real_pass():
    tr = TorchTrainer(TinyPolicy(32, 8), TrainerConfig(speed_coef=0.5))
    shaped = tr._shaped_rewards([_rec("p", 1.0, timings={"active_s": 2.0}), _rec("q", 1.0, timings={"active_s": 2.0}),
                                 _rec("partial", 0.18, timings={"active_s": 0.1})], ["g", "g", "g"])
    assert shaped[2] == pytest.approx(0.18)  # a partial-credit failure earns no speed bonus


# ---- KL (G4 #3) -----------------------------------------------------------------------------------------------------

def test_G4_3_kl_is_clamped_and_its_gradient_is_the_reverse_kl():
    tr = TorchTrainer(TinyPolicy(32, 8), TrainerConfig(kl_coef=1.0))
    from crucible.train.trainer import _Item

    it = _Item([1], [2, 3], [-1.0, -1.0], 0.0, 0, ref=[-0.01, -2.0], ref_mask=[1, 1], kl=1.0)
    lp = torch.tensor([[-30.0, -1.0]], dtype=torch.float64, requires_grad=True)
    out = tr._extra_terms([it], [0], lp, torch.ones(1, 2))
    assert float(out.max()) <= 10.0 + 1e-9  # NeMo-RL clamp; unclamped k3 was 1e13
    out.sum().backward()
    assert lp.grad[0, 1] == pytest.approx(-1.0 - (-2.0))  # d/dlp of the reverse-KL surrogate: (log π − log π_ref)


# ---- policy logits and weights (H1 V14, V15, V5, G2 H2) ----------------------------------------------------------------

def test_H1_V14_hf_head_applies_the_models_logit_softcap_and_scales():
    from types import SimpleNamespace

    from crucible.train.hf import _post_head

    x = torch.tensor([[60.0, -3.0]])
    assert torch.allclose(_post_head(x, SimpleNamespace(final_logit_softcapping=30.0)), torch.tanh(x / 30) * 30)
    assert torch.allclose(_post_head(x, SimpleNamespace(logits_scaling=4.0)), x / 4)
    assert torch.allclose(_post_head(x, SimpleNamespace(logit_scale=0.5)), x * 0.5)


def test_H1_V15_mxfp4_rounds_half_to_even():
    from crucible.train.weights import qdq_mxfp4

    # block amax 6 → scale 1: exact E2M1 midpoints
    x = torch.tensor([[6.0, 0.75, 1.75, 3.5, 5.0, 0.25, 2.5] + [0.0] * 25])
    q = qdq_mxfp4(x)[0, :7].tolist()
    assert q == [6.0, 1.0, 2.0, 4.0, 4.0, 0.0, 2.0]


def test_G2_H2_qdq_master_weights_accumulate_updates_smaller_than_the_grid():
    from crucible.train.weights import QDQ

    class Wrap(torch.nn.Module):  # HFPolicy-like: parameters carry a "model." prefix the published names do not
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.experts = torch.nn.Linear(32, 4, bias=False)

        def state_dict(self, *a, **k):
            return {k2.removeprefix("model."): v for k2, v in super().state_dict(*a, **k).items()}

    torch.manual_seed(0)
    w = Wrap()
    q = QDQ("mxfp4", model=w)
    pub = q(w.state_dict())
    assert torch.equal(pub["experts.weight"], w.model.experts.weight.detach())  # trainer and engine on one grid
    start = pub["experts.weight"].clone()
    for _ in range(200):  # many updates, each far below half a grid step
        with torch.no_grad():
            w.model.experts.weight += 1e-3
        pub = q(w.state_dict())
    assert (pub["experts.weight"] != start).float().mean() > 0.5  # the master moved 0.2: the grid follows


# ---- vision (H1 V6) --------------------------------------------------------------------------------------------------

def test_H1_V6_image_records_are_refused_by_a_text_only_policy():
    t = _turn(media=[{"ref": "sha256:" + "0" * 64, "offset": 0, "length": 1}])
    tr = TorchTrainer(TinyPolicy(32, 8), TrainerConfig())
    with pytest.raises(ValueError, match="images"):
        tr.step([_rec("a", 1.0, [t]), _rec("b", 0.0)], ["g", "g"])


# ---- new objectives the sweeps found missing (G4 #8, G5, G1 #10-11) ---------------------------------------------------

def test_G4_8_ipo_masks_by_absolute_probability_move():
    logp = torch.log(torch.tensor([[0.5, 0.9]], dtype=torch.float64)).requires_grad_(True)
    roll = torch.log(torch.tensor([[0.45, 0.4]], dtype=torch.float64))  # moves 0.05 and 0.5
    t = correct(Correction("ipo", {"eps": 0.3}), logp=logp, rollout_logp=roll, adv=torch.ones(1, 2, dtype=torch.float64), mask=torch.ones(1, 2))
    assert t.kept.tolist() == [[True, False]]
    t.objective.sum().backward()
    assert float(logp.grad[0, 0]) == pytest.approx(0.5 / 0.45)


def test_G5_sequence_level_tis_and_mask():
    logp = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    roll = torch.tensor([[-0.5, -0.5], [0.0, 0.0]], dtype=torch.float64)  # ρ = e^1 for row 0, 1 for row 1
    kw = {"logp": logp, "rollout_logp": roll, "old_logp": logp.detach(), "adv": torch.ones(2, 2, dtype=torch.float64), "mask": torch.ones(2, 2),
          "seq_ids": torch.tensor([0, 1])}
    t = correct(Correction("tis", {"level": "sequence", "C": 2.0}), **kw)
    assert t.objective[0, 0].item() == pytest.approx(2.0) and t.objective[1, 0].item() == pytest.approx(1.0)
    m = correct(Correction("icepop", {"level": "sequence", "alpha": 0.5, "beta": 2.0}), **kw)
    assert m.kept.tolist() == [[False, False], [True, True]]


def test_G1_11_cispo_unified_mask_recovers_ppo_trust_region():
    logp = torch.log(torch.tensor([[0.9, 0.1]], dtype=torch.float64)).requires_grad_(True)
    old = torch.log(torch.tensor([[0.5, 0.5]], dtype=torch.float64))
    t = correct(Correction("cispo", {"mask": True}), logp=logp, rollout_logp=old, old_logp=old, adv=torch.tensor([[1.0, -1.0]], dtype=torch.float64),
                mask=torch.ones(1, 2))
    assert t.kept.tolist() == [[False, False]]  # A>0 past 1+ε, A<0 below 1−ε: both outside PPO's region


def test_G5_unbiased_std_and_reinforce_baseline():
    from crucible.train.objectives import group_advantages

    a = group_advantages([1, 0, 0, 0], ["g"] * 4, std_normalize=True, std_ddof=1)
    assert a[0] == pytest.approx(0.75 / math.sqrt(0.25), rel=1e-5)  # TRL / verl: Bessel std 0.5
    b = group_advantages([1, 0, 1, 1], ["g", "g", "h", "h"], estimator="reinforce_pp_baseline")
    assert b[2] == b[3] == 0.0 and b[0] == -b[1] != 0


def test_G1_10_dapo_overlong_shaping_and_filter():
    long = _rec("long", 1.0, [_turn(comp=tuple(range(3, 3 + 10)))], truncated=True)
    tr = TorchTrainer(TinyPolicy(32, 8), TrainerConfig(overlong={"l_max": 8, "l_cache": 4}, overlong_filter=True))
    shaped = tr._shaped_rewards([long, _rec("s", 1.0)], ["g", "g"])
    assert shaped == [0.0, 1.0]  # 10 > l_max: −1; 3 ≤ 4: free
    items = tr._items([long, _rec("s", 1.0)], ["g", "g"])
    assert [it.seq for it in items] == ["s"]  # the truncated rollout is in the baseline, out of the loss


def test_G5_D9_micro_batch_budget_counts_padding():
    from crucible.train.batching import micro_batches

    L = [100] + [10] * 9  # raw tokens 190 fit one 200 budget; padded to the longest row they are 1000
    mbs = micro_batches(L, list(range(10)), 200)
    assert all(max(L[i] for i in mb.indices) * len(mb.indices) <= 200 for mb in mbs) and len(mbs) == 2


# ---- data parallelism over real gloo processes (H4 V1, V2, V12, G5 D1, G4 #2, G2 H8) -----------------------------------

WORKER = Path(__file__).with_name("_m12_dp_worker.py")


def _run(scenario: str, tmp_path: Path, world: int, port: int) -> list[dict]:
    env = {**os.environ, "WORLD_SIZE": str(world), "M12_PORT": str(port),
           "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")}
    procs = [subprocess.Popen([sys.executable, str(WORKER), scenario, str(tmp_path)], env={**env, "RANK": str(r)}, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for r in range(world)]
    errs = []
    for p in procs:
        try:
            _, err = p.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            p.kill()
            _, err = p.communicate()
            err = "TIMEOUT (hang)\n" + err
        if p.returncode:
            errs.append(err[-2000:])
    assert not errs, errs
    return [json.loads((tmp_path / f"r{r}.json").read_text()) for r in range(world)]


def test_H4_V1_V2_dp_with_an_empty_rank_and_a_rank_local_parameter(tmp_path):
    r = _run("dp", tmp_path, world=2, port=29581)[0]
    for k in ("empty_rank", "side_param", "seq_mean", "prompt_mean_ppo"):
        assert r[k]["max_abs_diff"] < 1e-6, (k, r[k])


def test_H4_V12_trainer_subgroup_source_is_a_group_rank(tmp_path):
    rs = _run("subgroup", tmp_path, world=3, port=29591)
    assert rs[1]["subgroup"]["w0"] == pytest.approx(rs[2]["subgroup"]["w0"])
