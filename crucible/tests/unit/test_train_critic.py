"""Critic mode (M12 checklist #16, #46–#48): token-level GAE advantages from a value model, skip-observation and
cross-segment wiring, truncation bootstrap, TTUR, frozen attention, and single-rollout (n = 1) learning where
group-relative advantages are identically zero."""

from __future__ import annotations

import copy

import torch

from crucible.train.critic import CriticConfig, HFValue, TinyValue
from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
from crucible.train.envs import CopyTask
from crucible.train.objectives import gae
from crucible.train.rollout import RolloutRecord, Turn
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import SchedulerConfig
from crucible.train.trainer import TorchTrainer, TrainerConfig
from crucible.train.weights import InprocPublisher


def _rec(reward=1.0, truncated=False, segments=(0, 0)):
    turns = [Turn(call_id=f"c{k}", actor="agent", segment=s, prompt_is_delta=k > 0 and s == segments[k - 1], prompt_token_ids=[1, 2, 3] if k == 0 else [4, 5],
                  completion_token_ids=[6 + k, 7, 8], logprobs=[-1.0] * 3, policy_version=0) for k, s in enumerate(segments)]
    return RolloutRecord(rollout_id="r", sample="r", task="t", harness="h", policy="p", reward=reward, truncated=truncated, turns=turns)


def test_trainer_advantages_equal_direct_gae_including_segments():
    for segs in ((0, 0), (0, 1)):
        critic = TinyValue(32, 8, seed=1)
        with torch.no_grad():
            critic.head.weight.normal_(0, 0.5)  # non-trivial values
        cc = CriticConfig(lam_policy=0.9)
        tr = TorchTrainer(TinyPolicy(32, 8), TrainerConfig(item_unit="call"), critic=critic, critic_cfg=cc)
        rec = _rec(segments=segs)
        items = tr._items([rec], ["g"])
        from crucible.train.rollout import full_prompts

        p = full_prompts(rec)
        with torch.no_grad():
            v = torch.cat([critic.token_values(p[t.call_id], t.completion_token_ids).double() for t in rec.turns])
        seg = torch.tensor([s for s in segs for _ in range(3)])
        multi = len(set(segs)) > 1
        rewards = torch.zeros(6, dtype=torch.float64)
        if not multi:
            rewards[-1] = 1.0
        kw = {"segments": seg, "shared_terminal": 1.0} if multi else {}
        want, _ = gae(rewards, v, torch.ones(6, dtype=torch.bool), gamma=1.0, lam=0.9, **kw)
        got = torch.tensor([a for it in items for a in it.adv])
        assert torch.allclose(got, want.float(), atol=1e-6), (segs, got, want)


def test_truncation_bootstraps_from_the_final_value():
    critic = TinyValue(32, 8, seed=1)
    with torch.no_grad():
        critic.head.bias.fill_(0.5)  # V ≡ 0.5
    cc = CriticConfig(lam_policy=1.0)
    full = TorchTrainer(TinyPolicy(32, 8), TrainerConfig(item_unit="call"), critic=critic, critic_cfg=cc)._items([_rec(reward=0.0)], ["g"])
    cut = TorchTrainer(TinyPolicy(32, 8), TrainerConfig(item_unit="call"), critic=critic, critic_cfg=cc)._items([_rec(reward=0.0, truncated=True)], ["g"])
    assert abs(full[-1].adv[-1] + 0.5) < 1e-6  # finished with 0: the last step lost the 0.5 the critic expected
    assert abs(cut[-1].adv[-1]) < 1e-6  # truncated is not failure: R̂ = V(s_T)


def test_ttur_and_frozen_attention_on_a_transformer_critic():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from crucible.train.hf import HFPolicy

    def _tiny_qwen():  # local copy: the suite runs from any rootdir (the lab container has no `tests` package on sys.path)
        torch.manual_seed(0)
        return Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                            num_key_value_heads=2, head_dim=8, max_position_embeddings=128, tie_word_embeddings=True))

    qwen = _tiny_qwen()
    critic = HFValue(copy.deepcopy(qwen), freeze_attention=True)
    attn0 = {n: q.detach().clone() for n, q in critic.backbone.named_parameters() if "self_attn" in n}
    mlp0 = {n: q.detach().clone() for n, q in critic.backbone.named_parameters() if "mlp" in n}
    tr = TorchTrainer(HFPolicy(qwen), TrainerConfig(lr=1e-3), critic=critic, critic_cfg=CriticConfig(updates_per_policy_update=3, lr=1e-2))
    recs = [_rec(reward=float(i % 2)).model_copy(update={"rollout_id": f"r{i}"}) for i in range(4)]
    for _ in range(2):
        m = tr.step(recs, [f"g{i}" for i in range(4)])
    assert tr.critic_updates == 6 and m["critic/updates"] == 6.0 and "critic/explained_variance" in m
    assert all(torch.equal(q, attn0[n]) for n, q in critic.backbone.named_parameters() if "self_attn" in n)  # frozen
    assert any(not torch.equal(q, mlp0[n]) for n, q in critic.backbone.named_parameters() if "mlp" in n)  # trained


async def _run(critic: bool, steps: int = 40):
    model = TinyPolicy(32, 48, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=64, decode_delay_s=0.001)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2), critic=TinyValue(32, 48, seed=2) if critic else None, critic_cfg=CriticConfig(lr=1e-2))
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=steps, max_turn_tokens=3), SchedulerConfig(n=1, B=16, max_staleness=2),
                 eng, tr, InprocPublisher(tr, eng), lambda: CopyTask(turns=2, symbols=8, latency_s=0.001), {"copy": [f"p{i}" for i in range(20000)]})
    return await rt.run()


async def test_single_rollout_learning_needs_the_critic():
    """n = 1: group-relative advantages are 0, so GRPO cannot move; the critic's baseline makes learning possible."""
    grpo = await _run(critic=False, steps=15)
    assert all(abs(s.metrics["train/loss"]) < 1e-12 for s in grpo.steps)  # zero advantage → zero gradient signal
    crit = await _run(critic=True)
    r = crit.series("reward/mean")
    ev = crit.series("critic/explained_variance")
    assert sum(r[-10:]) / 10 > sum(r[:5]) / 5 + 0.08, (r[:5], r[-10:])
    assert max(ev[-10:]) > ev[0]  # the critic learns to explain returns


def test_critic_advantages_do_not_collide_across_rollouts_with_the_same_call_ids():
    """World episodes all have call ids agent:c1, agent:c2...: per-token advantages are keyed by (rollout, call), so one
    record's GAE never overwrites another's."""
    tr = TorchTrainer(TinyPolicy(32, 8, seed=0), TrainerConfig(item_unit="call"), critic=TinyValue(32, 8, seed=1), critic_cfg=CriticConfig())
    a, b = _rec(reward=1.0), _rec(reward=0.0)
    b = b.model_copy(update={"rollout_id": "r2", "sample": "r2"})
    assert [t.call_id for t in a.turns] == [t.call_id for t in b.turns]
    adv = tr._gae_advantages([a, b])
    assert len(adv) == 2 * len(a.turns)
    items = tr._items([a, b], ["g1", "g2"])
    first = [x for it in items[: len(a.turns)] for x in it.adv]
    second = [x for it in items[len(a.turns):] for x in it.adv]
    assert first != second and first[-1] > second[-1]  # the rewarded rollout keeps its own, larger advantage
