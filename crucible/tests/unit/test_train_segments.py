"""Segment-level trainer items (m13 B-1): one causal sequence per (rollout, actor, segment), loss only on sampled tokens.
The loss and the gradient equal the per-call items (every call re-encoding its full prompt), while forward tokens drop
from O(turns²) to O(turns); works for group advantages, critic GAE and compaction segments."""

from __future__ import annotations

import copy

import pytest
import torch

from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.train.critic import CriticConfig, TinyValue
from crucible.train.engine import TinyPolicy
from crucible.train.mismatch import Correction
from crucible.train.trainer import TorchTrainer, TrainerConfig


def _rec(i: int, turns: int = 6, segments=None) -> RolloutRecord:
    g = torch.Generator().manual_seed(i)
    segs = segments or [0] * turns
    out = []
    for k in range(turns):
        new_seg = k == 0 or segs[k] != segs[k - 1]
        prompt = torch.randint(4, 60, (12 if new_seg else 5,), generator=g).tolist()  # first call: full prompt; later: the observation delta
        comp = torch.randint(4, 60, (int(torch.randint(2, 6, (1,), generator=g)),), generator=g).tolist()
        out.append(Turn(sampling={"temperature": 1.0}, call_id=f"r{i}:c{k}", actor="agent", segment=segs[k], prompt_is_delta=not new_seg, prompt_token_ids=prompt,
                        completion_token_ids=comp, logprobs=[-2.0] * len(comp), policy_version=0, advantage=0.1 * k))
    return RolloutRecord(rollout_id=f"r{i}", sample="s", task="t", harness="h", policy="p", reward=float(i % 2), turns=out)


@pytest.mark.parametrize("critic", [False, True])
@pytest.mark.parametrize("segmented", [False, True])
def test_segment_items_equal_call_items(critic, segmented):
    base = TinyPolicy(64, 16, seed=7)
    segs = [0, 0, 0, 1, 1, 1] if segmented else None
    recs = [_rec(i, segments=segs) for i in range(4)]
    groups = ["g", "g", "h", "h"]
    out = {}
    for unit in ("call", "segment"):
        m = copy.deepcopy(base)
        cr = TinyValue(64, 16, seed=8) if critic else None
        tr = TorchTrainer(m, TrainerConfig(lr=1e-2, item_unit=unit, turn_weight=0.5, token_budget=1 << 20, correction=Correction("ppo")),
                          critic=cr, critic_cfg=CriticConfig(lam_policy=0.9))
        met = tr.step(recs, groups)
        out[unit] = (met, {n: q.grad.detach().clone() for n, q in m.named_parameters() if q.grad is not None})
    (mc, gc), (ms, gs) = out["call"], out["segment"]
    assert abs(mc["train/loss"] - ms["train/loss"]) < 1e-6 and mc["train/tokens"] == ms["train/tokens"]
    assert max(float((gc[k] - gs[k]).abs().max()) for k in gc) < 1e-6
    assert ms["train/forward_tokens"] < mc["train/forward_tokens"] and ms["train/items"] < mc["train/items"]


def test_forward_tokens_are_linear_in_turns():
    def fwd(unit: str, turns: int) -> float:
        tr = TorchTrainer(TinyPolicy(64, 8), TrainerConfig(item_unit=unit, token_budget=1 << 20))
        return tr.step([_rec(1, turns=turns), _rec(2, turns=turns)], ["g", "g"])["train/forward_tokens"]

    seg = [fwd("segment", t) for t in (10, 20, 40)]
    call = [fwd("call", t) for t in (10, 20, 40)]
    assert seg[2] / seg[1] < 2.3 and call[2] / call[1] > 3.0  # doubling turns: ~2× vs ~4×
