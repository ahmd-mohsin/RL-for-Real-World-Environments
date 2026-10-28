"""M12-infra batch B: estimators (ReMax, REINFORCE++), reward-to-go credit (MiniMax M2), speed and latency shaping (MiniMax
M2.5, Kimi PARL critical steps, DeepSeek DAG critical path), frozen sub-agents (PARL) in the reference trainer,
entropy-adaptive mimo4 bounds (MiMo), group operators (MiMo GRS / GAR, StepFun pairwise GenRM, tournaments, hack zeroing),
QDQ to the rollout grid on both sides (MiMo MXFP4, FP8), and value pretraining (Seed / SAO)."""

from __future__ import annotations

import math

import pytest
import torch

from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.train.credit import critical_steps, dag_critical_path
from crucible.train.critic import TinyValue, pretrain_value
from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
from crucible.train.group_ops import as_group_fn, bt_pairwise, comparisons, compose, gar, grs, tournament, zero_hacks
from crucible.train.mismatch import Correction
from crucible.train.objectives import group_advantages, reward_to_go, speed_bonus
from crucible.train.trainer import TorchTrainer, TrainerConfig
from crucible.train.weights import FP4_E2M1, QDQ, InprocPublisher, qdq_fp8_e4m3, qdq_mxfp4


def _turn(k, actor="agent", deps=None, step_reward=None, n=3):
    return Turn(sampling={"temperature": 1.0}, call_id=f"c{k}", actor=actor, prompt_is_delta=k > 0, prompt_token_ids=[1, 2] if k == 0 else [3], completion_token_ids=[4 + k % 20] * n,
                logprobs=[-1.0] * n, policy_version=0, depends_on=deps, step_reward=step_reward)


def _rec(rid, reward, turns=None, **kw):
    return RolloutRecord(rollout_id=rid, sample=rid, task="t", harness="h", policy="p", reward=reward, turns=turns or [_turn(0)], **kw)


# ----------------------------------------------------------------------------------------------- estimators and credit math
def test_estimators():
    r, g = [1.0, 0.0, 1.0, None, 0.5, 0.5], ["a", "a", "a", "a", "b", "b"]
    assert group_advantages(r, g, estimator="remax", baselines={"a": 0.0, "b": 1.0}) == [1.0, 0.0, 1.0, 0.0, -0.5, -0.5]
    rpp = group_advantages(r, g, estimator="reinforce_pp")
    live = [x for x, y in zip(rpp, r, strict=True) if y is not None]
    assert abs(sum(live)) < 1e-9 and abs(sum(x * x for x in live) / len(live) - 1.0) < 1e-4 and rpp[3] == 0.0  # global batch normalization
    with pytest.raises(ValueError, match="greedy baseline"):
        group_advantages(r, g, estimator="remax")
    with pytest.raises(ValueError, match="estimator"):
        group_advantages(r, g, estimator="vtrace")


def test_reward_to_go_speed_and_latency():
    assert reward_to_go([0.1, None, -0.2], 1.0) == pytest.approx([0.9, 0.8, 0.8])
    assert speed_bonus(10, 10) == 0.0 and speed_bonus(5, 10) == 0.5 and speed_bonus(30, 10) == -1.0 and speed_bonus(20, 10, kind="log") == -1.0
    # main, sub1, sub1, sub2 | main, sub1: stages (1 + max(2, 1)) + (1 + 1) = 5 (Kimi PARL)
    rec = _rec("r", 1.0, [_turn(0, "main"), _turn(1, "s1"), _turn(2, "s1"), _turn(3, "s2"), _turn(4, "main"), _turn(5, "s1")])
    assert critical_steps(rec) == 5 and critical_steps(rec, "main") == 5
    # DAG: c0 -> {c1 -> c2, c3} -> c4 : longest chain c0, c1, c2, c4 = 4 calls
    dag = _rec("d", 1.0, [_turn(0, "main"), _turn(1, "s1", ["c0"]), _turn(2, "s1", ["c1"]), _turn(3, "s2", ["c0"]), _turn(4, "main", ["c2", "c3"])])
    assert dag_critical_path(dag) == 4.0 and dag_critical_path(dag, weight="tokens") == 12.0


def _trainer(**cfg):
    return TorchTrainer(TinyPolicy(32, 8, seed=0), TrainerConfig(item_unit="call", **cfg))


def test_trainer_shaping_reward_to_go_and_frozen_actors():
    fast = _rec("fast", 1.0, timings={"active_s": 5.0})
    slow = _rec("slow", 1.0, timings={"active_s": 20.0})
    fail = _rec("fail", 0.0, timings={"active_s": 1.0})
    tr = _trainer(speed_coef=0.2)
    items = tr._items([fast, slow, fail], ["g"] * 3)
    assert fast.reward_components["shaping/speed"] > 0 > slow.reward_components["shaping/speed"]
    assert "shaping/speed" not in fail.reward_components  # gated: a fast failure earns no speed bonus
    assert items[0].adv > items[1].adv

    multi = _rec("m", 1.0, [_turn(0, "main"), _turn(1, "s1"), _turn(2, "s1"), _turn(3, "main")])
    single = _rec("s", 1.0, [_turn(0, "main"), _turn(1, "main")])
    tr = _trainer(latency_coef=0.1)
    tr._items([multi, single], ["g", "g"])
    assert multi.reward_components["shaping/latency"] == pytest.approx(-0.1 * critical_steps(multi))

    rtg = [_rec("a", 1.0, [_turn(0, step_reward=0.5), _turn(1, step_reward=-0.5)]), _rec("b", 0.0, [_turn(0, step_reward=0.0), _turn(1, step_reward=0.0)])]
    items = _trainer(credit_unit="reward_to_go")._items(rtg, ["g", "g"])
    # G: a = [1.0, 0.5], b = [0, 0]; baseline = mean of G_0 = 0.5 -> a: [0.5, 0.0], b: [-0.5, -0.5]
    assert [it.adv[0] for it in items] == pytest.approx([0.5, 0.0, -0.5, -0.5])

    orch = _rec("o", 1.0, [_turn(0, "orchestrator"), _turn(1, "worker"), _turn(2, "orchestrator")])
    assert len(_trainer()._items([orch], ["g"])) == 3
    frozen = _trainer(frozen_actors=("worker",))
    assert len(frozen._items([orch], ["g"])) == 2  # the sub-agent's call carries no loss (PARL)
    seg = TorchTrainer(TinyPolicy(32, 8, seed=0), TrainerConfig(frozen_actors=("worker",)))._items([orch], ["g"])
    assert sum(sum(it.mask) for it in seg if it.mask) == 6  # segment packing trains only the orchestrator's 2 × 3 tokens
    unfrozen = TorchTrainer(TinyPolicy(32, 8, seed=0), TrainerConfig())._items([orch], ["g"])
    assert sum(sum(it.mask) for it in unfrozen if it.mask) == 9


def test_entropy_adaptive_bounds():
    tr = _trainer(correction=Correction("mimo4"), entropy_adaptive={"low": 2.0, "high": 3.0, "rate": 0.25})
    low = [_rec(f"r{i}", float(i % 2)) for i in range(4)]  # behaviour logprob −1 → entropy 1 < low
    m = tr.step(low, ["g", "g", "h", "h"])
    p = tr.bands  # trainer state (saved by state.capture), never the configuration (async-RL audit: that broke resume)
    assert tr.cfg.correction.params == {} and tr._correction().params == p
    assert m["entropy_adaptive/entropy"] == pytest.approx(1.0)
    assert p["pos_high"] == pytest.approx(6.25) and p["pos_low"] == pytest.approx(0.16) and p["neg_high"] == pytest.approx(4.0) and p["neg_low"] == pytest.approx(0.25)
    for _ in range(40):
        tr.step(low, ["g", "g", "h", "h"])
    assert p["pos_high"] == 20.0 and p["pos_low"] == 0.05 and p["neg_high"] == 1.01  # clamped bands


# ----------------------------------------------------------------------------------------------- group operators
async def test_gar_conserves_positive_mass_and_caps():
    members = [_rec("p1", 1.0), _rec("p2", 1.0), _rec("f1", 0.0), _rec("f2", 0.0)]
    q = {"p1": 1.0, "p2": 0.5}
    out = await gar(lambda m: q[m.rollout_id])("g", members)
    a = [m.advantage for m in out]
    assert a == pytest.approx([2 / 3, 1 / 3, -0.5, -0.5]) and sum(a[:2]) == pytest.approx(1.0)  # Σ_P A' = Σ_P A
    capped = await gar(lambda m: q[m.rollout_id], lambda_max=1.2)("g", [_rec("p1", 1.0), _rec("p2", 1.0), _rec("f1", 0.0), _rec("f2", 0.0)])
    ca = [m.advantage for m in capped]
    assert abs(sum(ca)) < 1e-12 and ca[0] > ca[1] > 0 > ca[2]  # re-centred when the cap binds
    same = await gar(lambda m: 0.5)("g", [_rec("x", 1.0), _rec("y", 1.0)])
    assert [m.advantage for m in same] == [0.0, 0.0]  # an all-pass group has no mixed outcome to redistribute


async def test_grs_hack_zeroing_bt_and_tournament():
    rub = {"good": (1.0, 0.9), "sloppy": (0.5, 0.5)}
    out = await compose(zero_hacks(lambda m: m.rollout_id == "leak"), grs(lambda m: rub.get(m.rollout_id)))(
        "g", [_rec("good", 1.0), _rec("sloppy", 1.0), _rec("leak", 1.0), _rec("fail", 0.0)])
    assert [m.reward for m in out] == pytest.approx([0.9, 0.25, 0.0, 0.0])
    assert out[2].reward_components["hack/zeroed"] == 1.0 and "grs/solution" not in out[3].reward_components

    async def judge(m, ref):
        return {"a": 0.9, "b": 0.4}[m.rollout_id]

    bt = await bt_pairwise(judge, reference="ref")("g", [_rec("a", 1.0), _rec("b", 1.0)])
    assert [m.reward for m in bt] == [0.9, 0.4]
    quality = {"x": 3, "y": 2, "z": 1}
    tour = await tournament(lambda a, b: 1.0 if quality[a.rollout_id] > quality[b.rollout_id] else 0.0)("g", [_rec(k, 1.0) for k in "xyz"])
    assert [m.reward for m in tour] == [1.0, 0.5, 0.0]
    assert comparisons(8, "bt_pairwise") == 8 and comparisons(8, "tournament") == 28


async def test_group_ops_in_the_runtime():
    from crucible.train.envs import CopyTask
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig

    model = TinyPolicy(32, 48, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=32, decode_delay_s=0.001)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2))
    rt = Runtime(RuntimeConfig(steps=4, max_turn_tokens=3), SchedulerConfig(n=4, B=2, max_staleness=2), eng, tr, InprocPublisher(tr, eng),
                 lambda: CopyTask(turns=2, symbols=8), {"copy": [f"p{i}" for i in range(400)]}, group_fn=as_group_fn(gar(lambda m: 0.7)))
    rep = await rt.run()
    assert sum(rep.series("group/graded")) > 0 and all(r.advantage is not None for r in rep.trained_records)


# ----------------------------------------------------------------------------------------------- QDQ and value pretraining
def test_qdq_grids():
    x = torch.randn(4, 70)
    q = qdq_mxfp4(x)
    assert torch.equal(qdq_mxfp4(q), q)  # idempotent: already on the grid
    blk = q[0, :32]
    scale = 2.0 ** (math.floor(math.log2(x[0, :32].abs().max().item())) - 2)
    assert set((blk.abs() / scale).tolist()) <= set(FP4_E2M1)
    f8 = qdq_fp8_e4m3(x)
    assert (f8 - x).abs().max() < 0.1 * x.abs().max() and not torch.equal(f8, x)


async def test_qdq_makes_trainer_and_engine_bit_identical():
    model = TinyPolicy(32, 16, seed=3)
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2))
    eng = EnginePool([TorchEngine(model, capacity=4)])
    qdq = QDQ("mxfp4", include=("*",), model=model)
    pub = InprocPublisher(tr, eng, post_update_transform=qdq)
    await pub.publish()
    served = eng.engines[0]._serve.state_dict() if hasattr(eng.engines[0], "_serve") else None
    assert qdq.applied
    for name, p in model.named_parameters():
        if name in qdq.applied:
            assert torch.equal(qdq_mxfp4(p.detach()), p.detach())  # the trainer's own weights sit on the grid
            if served is not None and name in served:
                assert torch.equal(served[name].to(p.dtype), p.detach())


def test_value_pretraining_fits_returns():
    recs = [_rec(f"r{i}", float(i % 2), [_turn(i % 2)]) for i in range(8)]
    v = TinyValue(32, 16, seed=1)
    out = pretrain_value(v, recs, steps=80, lr=5e-2)
    assert out["loss"][-1] < 0.3 * out["loss"][0] and out["explained_variance"] > 0.5
    with pytest.raises(ValueError, match="scored"):
        pretrain_value(v, [_rec("m", None)])
