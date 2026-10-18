"""Crucible's credit through slime's hooks (swarm-pipeline/00 §11 item 4; 03 §4.3).

slime has no step credit: every segment gets the full outcome (``slime/agent/trajectory.py:417-419``) and grpo broadcasts
one scalar per sample (``slime/backends/megatron_utils/loss.py:784-788``). Three hooks carry Crucible's rules instead:

- ``reward_post_process`` (``--custom-reward-post-process-path``, called by ``BatchBuilder._post_process_rewards``,
  ``batch_builder.py:210-212``): the sequence advantage per **unique trajectory** (slime's own grouping rule,
  ``:213-255``), Dr.GRPO style ``R − mean`` over the live members of the GRPO group (no std normalisation, as slime's
  coding recipe runs it: ``--disable-grpo-std-normalization``), then the honesty term composed exactly as
  ``factory.honesty.compose_advantage`` does (a failure never turns positive, a success is floored at 0). Removed samples
  are in no statistic and get 0.
- ``convert`` (``--custom-convert-samples-to-train-data-path``, ``batch_builder.py:263-264``): slime's own conversion
  (``BatchBuilder.convert`` with the post-process above; a mirror of ``:266-341`` when slime is absent) plus one new
  per-token key, ``turn_advantages``: each turn's process credit on its own completion tokens, from the per-turn ΔΦ the
  generate function attached, through M6's group rule (``credit.compose.turn_credit``: centred within the episode,
  bounded by the outcome advantage, none on gate-failed or hacked episodes, none without outcome spread). This is
  ``train.credit.turn_credit_map`` computed from slime samples instead of rollout records.
- ``advantage`` (``--custom-advantage-function-path``, ``loss.py:779-782``): ``advantages[i] = A_seq[i]`` broadcast over
  the response tokens + ``turn_advantages[i]`` (sliced for context parallelism the way slime slices log probs,
  ``actor.py:260-275``). The key survives the data-parallel split only with ``batch_builder_turn_advantages.patch``
  (``batch_builder.py:467-487`` forwards a fixed field list); without it this function raises instead of silently
  training outcome-only.

``eta`` (the turn-credit weight): ``args.crucible_turn_eta`` / ``CRUCIBLE_SLIME_TURN_ETA``, default 0.1 (``credit.compose``).
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Sequence
from typing import Any

from crucible.credit.compose import Episode, turn_credit
from crucible.train.bridges.slime.types import flat, is_masked, rollout_key

PATCH_FILE = "batch_builder_turn_advantages.patch"


def eta_of(args: Any) -> float:
    v = getattr(args, "crucible_turn_eta", None)
    if v is None:
        v = os.environ.get("CRUCIBLE_SLIME_TURN_ETA")
    return float(v) if v not in (None, "") else 0.1


def compose_advantage(outcome_adv: float, term: float) -> float:
    """``factory.honesty.compose_advantage``, mirrored here because ``crucible.train`` sits below the factory (pinned equal
    by the P6 equivalence test)."""
    if term > 0:
        raise ValueError("the honesty term is a penalty (≤ 0)")
    a = outcome_adv + term
    return max(a, 0.0) if outcome_adv > 0 else min(a, 0.0)


def _crucible(s: Any) -> dict[str, Any]:
    return (getattr(s, "metadata", None) or {}).get("crucible") or {}


def group_credit(samples: Sequence[Any], *, eta: float) -> dict[Any, dict[str, Any]]:
    """Per trajectory key: ``{"seq": composed sequence advantage, "turns": [per-turn credit], "reward", "masked"}``.

    Trajectories are the unique ``rollout_key``s of ``samples`` (segments of one rollout share it); groups are
    ``group_index`` (all in one group when unset)."""
    segs = flat(samples)
    traj: dict[Any, Any] = {}
    for pos, s in enumerate(segs):
        traj[rollout_key(s, pos)] = s  # the last segment carries the rollout's metadata
    groups: dict[Any, list[Any]] = {}
    for k, s in traj.items():
        groups.setdefault(getattr(s, "group_index", None), []).append(k)
    out: dict[Any, dict[str, Any]] = {}
    for keys in groups.values():
        live = [k for k in keys if not is_masked(traj[k])]
        mu = sum(float(traj[k].reward) for k in live) / len(live) if live else 0.0
        eps = []
        for k in keys:
            s = traj[k]
            c = _crucible(s)
            masked = is_masked(s)
            deltas = [float(t.get("delta_phi") or 0.0) for t in c.get("turn_credit") or []]
            gates = c.get("gates") or {}
            eps.append(Episode(reward=None if masked else float(s.reward), phis=list(itertools.accumulate(deltas)), gated=bool(gates) and not all(gates.values()),
                               hack=float(c.get("hack_confirmed") or 0.0) > 0))
        rows = turn_credit(eps, eta=eta)
        for k, row in zip(keys, rows, strict=True):
            s = traj[k]
            if is_masked(s):
                out[k] = {"seq": 0.0, "turns": [0.0] * len(row), "reward": None, "masked": True}
                continue
            a = float(s.reward) - mu
            term = min(0.0, float(_crucible(s).get("honesty_term") or 0.0))
            out[k] = {"seq": compose_advantage(a, term), "turns": list(row), "reward": float(s.reward), "masked": False, "outcome_adv": a, "honesty_term": term}
    return out


def reward_post_process(args: Any, samples: Any) -> tuple[list[float], list[float]]:
    """``(raw_rewards, rewards)`` aligned with the flat samples (see the module doc)."""
    segs = flat(samples)
    gc = group_credit(segs, eta=eta_of(args))
    raw = [0.0 if is_masked(s) else float(s.get_reward_value(args)) for s in segs]
    adv = [gc[rollout_key(s, i)]["seq"] for i, s in enumerate(segs)]
    return raw, adv


def turn_token_advantages(args: Any, samples: Any) -> list[list[float]]:
    """Per sample, one float per response token: the turn's credit on its completion span, 0 elsewhere."""
    segs = flat(samples)
    gc = group_credit(segs, eta=eta_of(args))
    out = []
    for i, s in enumerate(segs):
        row = [0.0] * int(s.response_length)
        g = gc[rollout_key(s, i)]
        mask = s.loss_mask or [1] * int(s.response_length)
        if not g["masked"]:
            for credit, t in zip(g["turns"], _crucible(s).get("turn_credit") or [], strict=False):
                a, b = t["span"]
                for j in range(a, min(b, len(row))):
                    row[j] = credit if mask[j] else 0.0
        out.append(row)
    return out


def _mirror_convert(args: Any, samples: list[Any]) -> dict[str, Any]:
    """slime ``BatchBuilder.convert`` (``batch_builder.py:259-341``) for the fields the coding bridge produces."""
    raw, rewards = reward_post_process(args, samples)
    rids = [s.rollout_id if s.rollout_id is not None else s.index for s in samples]
    seen = {r for r in rids if r is not None}
    tmp = 0
    for i in range(len(rids)):
        if rids[i] is None:
            while tmp in seen:
                tmp += 1
            rids[i] = tmp
            seen.add(tmp)
    masks = []
    for s in samples:
        if s.loss_mask is None:
            s.loss_mask = [1] * s.response_length
        if len(s.loss_mask) != s.response_length:
            raise ValueError(f"loss mask length {len(s.loss_mask)} != response length {s.response_length}")
        if s.remove_sample:
            s.loss_mask = [0] * s.response_length
        masks.append(s.loss_mask)
    tot: dict[Any, int] = {}
    for r, m in zip(rids, masks, strict=True):
        tot[r] = tot.get(r, 0) + sum(m)
    data: dict[str, Any] = {"tokens": [s.tokens for s in samples], "response_lengths": [s.response_length for s in samples], "rewards": rewards,
                            "raw_reward": raw, "truncated": [1 if s.status.value == "truncated" else 0 for s in samples],
                            "sample_indices": [s.index for s in samples], "rollout_ids": rids, "loss_masks": masks,
                            "rollout_mask_sums": [tot[r] for r in rids]}
    if any(s.metadata and "raw_reward" in s.metadata for s in samples):
        data["raw_reward"] = [s.metadata["raw_reward"] if s.metadata and "raw_reward" in s.metadata else s.reward for s in samples]
    if samples and samples[0].rollout_log_probs is not None:
        data["rollout_log_probs"] = [s.rollout_log_probs for s in samples]
    return data


def convert(args: Any, samples: Any) -> dict[str, Any]:
    segs = flat(samples)
    try:
        from slime.data.batch_builder import BatchBuilder
    except ImportError:
        data = _mirror_convert(args, segs)
    else:
        class _Shim:  # what BatchBuilder.convert reads from self
            pass

        shim = _Shim()
        shim.args = args
        shim.custom_convert_samples_to_train_data_func = None
        shim.custom_reward_post_process_func = reward_post_process
        shim._post_process_rewards = lambda s: reward_post_process(args, s)
        data = BatchBuilder.convert(shim, segs)
    data["turn_advantages"] = turn_token_advantages(args, segs)
    return data


def advantage(args: Any, rollout_data: dict[str, Any]) -> None:
    """Fill ``advantages`` and ``returns``: the sequence advantage on every response token + the turn credit."""
    import torch

    if "turn_advantages" not in rollout_data:
        if getattr(args, "crucible_allow_no_turn_credit", False):
            rollout_data["turn_advantages"] = None
        else:
            raise KeyError(f"rollout_data has no 'turn_advantages': apply {PATCH_FILE} (slime/data/batch_builder.py:467-487 forwards a fixed field "
                           "list), or set --crucible-allow-no-turn-credit to train outcome-only")
    kl = rollout_data["kl"]
    rewards = rollout_data["rewards"]
    turn = rollout_data.get("turn_advantages")
    cp = int(getattr(args, "context_parallel_size", 1) or 1)
    advs = []
    for i, k in enumerate(kl):
        a = torch.full_like(k, float(rewards[i]), dtype=torch.float32)
        if turn is not None:
            t = turn[i] if isinstance(turn[i], torch.Tensor) else torch.tensor(turn[i], dtype=torch.float32)
            if cp > 1:  # slice like the log probs (actor.py:260-275)
                from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

                t = slice_log_prob_with_cp(t, rollout_data["total_lengths"][i], rollout_data["response_lengths"][i])
            if t.numel() != a.numel():
                raise ValueError(f"sample {i}: {t.numel()} turn advantages for {a.numel()} response tokens")
            a = a + t.to(a.device, dtype=a.dtype)
        advs.append(a)
    rollout_data["advantages"] = advs
    rollout_data["returns"] = [a.clone() for a in advs]
