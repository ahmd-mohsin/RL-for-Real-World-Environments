"""The training backend contract and the reference torch trainer (M12).

``TrainerBackend.step(records, groups) -> metrics`` consumes one global batch of ``crucible.rollout/v1`` records (the
selected groups), and ``state_dict()`` exposes the weights the publisher pushes to the engines. Bridges to verl /
prime-rl / TRL implement the same contract (``bridges/``); this reference trainer implements it directly with Crucible's
own math so every knob is testable end to end:

1. advantages: group-relative (``objectives.group_advantages``; masked rollouts excluded) + optional M6 turn credit;
2. per-call training items: full prompt (reconstructed from delta storage) + completion tokens + behaviour logprobs;
3. placement: DP balancing and token-budget micro batches (``batching``); with ``token_mean_global`` each micro batch
   is scaled by its share of the global token count, so the gradient equals the full-batch gradient;
4. π_θold: recomputed with the pre-update weights only when the correction needs it;
5. the correction (``mismatch.correct``) gives per-token objectives; the loss is −aggregate; gradient clipping; one or
   more updates per batch; optional optimizer reset at publish (GLM).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from crucible.train.batching import balance_partitions, micro_batches
from crucible.train.mismatch import NEEDS_OLD, Correction, correct
from crucible.train.objectives import aggregate, group_advantages
from crucible.train.rollout import RolloutRecord, full_prompts


class TrainerBackend(Protocol):
    def step(self, records: Sequence[RolloutRecord], groups: Sequence[str]) -> dict[str, float]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def reset_optimizer(self) -> None: ...


@dataclass
class TrainerConfig:
    lr: float = 3e-3
    correction: Correction = field(default_factory=Correction)
    loss_agg: str = "token_mean_global"
    estimator: str = "mean"
    std_normalize: bool = False
    turn_weight: float = 0.0
    token_budget: int = 4096
    dp: int = 1
    grad_clip: float = 1.0
    updates_per_batch: int = 1
    temperature: float = 1.0


@dataclass
class _Item:
    prompt: list[int]
    completion: list[int]
    rollout_lp: list[float]
    adv: float
    prompt_index: int


class TorchTrainer:
    def __init__(self, model: torch.nn.Module, cfg: TrainerConfig | None = None):
        self.model, self.cfg = model, cfg or TrainerConfig()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        self.updates = 0

    def reset_optimizer(self) -> None:
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def _items(self, records: Sequence[RolloutRecord], groups: Sequence[str]) -> list[_Item]:
        advs = group_advantages([r.reward if not r.mask_sample else None for r in records], groups, std_normalize=self.cfg.std_normalize,
                                estimator=self.cfg.estimator)
        pids = {g: i for i, g in enumerate(dict.fromkeys(groups))}
        items = []
        for r, a, g in zip(records, advs, groups, strict=True):
            if r.mask_sample or r.reward is None:
                continue
            prompts = full_prompts(r)
            for t in r.turns:
                if not t.completion_token_ids:
                    continue
                items.append(_Item(prompts[t.call_id], list(t.completion_token_ids), list(t.logprobs or []), a + self.cfg.turn_weight * (t.advantage or 0.0), pids[g]))
        return items

    def _logprobs(self, items: list[_Item], idx: list[int]) -> list[torch.Tensor]:
        if hasattr(self.model, "batch_token_logprobs"):  # one forward per micro batch (HF policies)
            return self.model.batch_token_logprobs([(items[i].prompt, items[i].completion) for i in idx], self.cfg.temperature)
        return [self.model.token_logprobs(items[i].prompt, items[i].completion, self.cfg.temperature) for i in idx]

    def _padded(self, items: list[_Item], idx: list[int], old: dict[int, torch.Tensor] | None):
        T = max(len(items[i].completion) for i in idx)
        rows = self._logprobs(items, idx)
        dev = rows[0].device
        roll = torch.zeros(len(idx), T, dtype=torch.float64, device=dev)
        adv = torch.zeros(len(idx), T, dtype=torch.float64, device=dev)
        mask = torch.zeros(len(idx), T, device=dev)
        old_t = torch.zeros(len(idx), T, dtype=torch.float64, device=dev)
        lp_rows = []
        for k, i in enumerate(idx):
            it = items[i]
            n = len(it.completion)
            lp_rows.append(torch.nn.functional.pad(rows[k].double(), (0, T - n)))
            roll[k, :n] = torch.tensor(it.rollout_lp[:n], dtype=torch.float64, device=dev)
            adv[k, :n] = it.adv
            mask[k, :n] = 1
            if old is not None:
                old_t[k, :n] = old[i].to(dev)
        return torch.stack(lp_rows), roll, adv, mask, (old_t if old is not None else None)

    def step(self, records: Sequence[RolloutRecord], groups: Sequence[str]) -> dict[str, float]:
        items = self._items(records, groups)
        if not items:
            return {"train/items": 0.0}
        lengths = [len(it.completion) for it in items]
        g_tokens = sum(lengths)
        old = None
        if self.cfg.correction.name in NEEDS_OLD and self.cfg.correction.old_logprob_source is None:
            with torch.no_grad():
                idx_all = list(range(len(items)))
                old = {i: r.double() for i, r in zip(idx_all, self._logprobs(items, idx_all), strict=True)}
        metrics: dict[str, float] = {}
        single = self.cfg.loss_agg != "token_mean_global"
        for _ in range(self.cfg.updates_per_batch):
            self.opt.zero_grad()
            total_loss = 0.0
            parts = balance_partitions(lengths, self.cfg.dp)
            mbs = [mb.indices for p in parts for mb in micro_batches(lengths, p, self.cfg.token_budget)] if not single else [list(range(len(items)))]
            agg_metrics: list[dict] = []
            for idx in mbs:
                lp, roll, adv, mask, old_t = self._padded(items, idx, old)
                terms = correct(self.cfg.correction, logp=lp, rollout_logp=roll, adv=adv, mask=mask, old_logp=old_t)
                if single:
                    J = aggregate(terms.objective, mask, self.cfg.loss_agg, prompt_ids=[items[i].prompt_index for i in idx])
                else:
                    J = (terms.objective * mask).sum() / g_tokens  # this micro batch's share of the global token mean
                loss = -J
                loss.backward()
                total_loss += float(loss.detach())
                agg_metrics.append(terms.metrics)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            self.updates += 1
            metrics = {"train/loss": total_loss, "train/items": float(len(items)), "train/tokens": float(g_tokens), "train/micro_batches": float(len(mbs))}
            for k in agg_metrics[0]:
                metrics[f"mismatch/{k}"] = sum(m.get(k, 0.0) for m in agg_metrics) / len(agg_metrics)
        return metrics
