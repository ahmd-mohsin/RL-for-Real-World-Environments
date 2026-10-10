"""The training backend contract and the reference torch trainer (M12).

``TrainerBackend.step(records, groups) -> metrics`` consumes one global batch of ``crucible.rollout/v1`` records (the
selected groups), and ``state_dict()`` exposes the weights the publisher pushes to the engines. Bridges to verl /
prime-rl / TRL implement the same contract (``bridges/``); this reference trainer implements it directly with Crucible's
own math so every knob is testable end to end:

1. advantages: group-relative (``objectives.group_advantages``; masked rollouts excluded) + optional M6 turn credit;
2. per-call training items: full prompt (reconstructed from delta storage) + completion tokens + behaviour logprobs;
3. placement: DP balancing and token-budget micro batches (``batching``) over **forward** tokens (prompt + completion:
   memory and compute), while the loss normalizer is **trained** tokens (completions); with ``token_mean_global`` each
   micro batch is scaled by its share of the global trained-token count, so the gradient equals the full-batch
   gradient; ``DistributedTorchTrainer`` runs one partition per DP rank and all-reduces the gradients;
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
    candidates: list[list[int]] | None = None  # top-p nucleus per token: π_θ is renormalized over it (checklist #37)


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
                if not t.candidates_valid():
                    raise ValueError(f"{t.call_id}: sampled with top_p={t.sampling.get('top_p')} but no complete candidate sets; the trainer cannot "
                                     "renormalize π_θ (record candidates or sample with top_p=1)")
                items.append(_Item(prompts[t.call_id], list(t.completion_token_ids), list(t.logprobs or []), a + self.cfg.turn_weight * (t.advantage or 0.0), pids[g],
                                   t.candidates if float(t.sampling.get("top_p", 1.0)) < 1.0 else None))
        return items

    def _logprobs(self, items: list[_Item], idx: list[int]) -> list[torch.Tensor]:
        if hasattr(self.model, "batch_token_logprobs"):  # one forward per micro batch (HF policies)
            return self.model.batch_token_logprobs([(items[i].prompt, items[i].completion) for i in idx], self.cfg.temperature,
                                                   candidates=[items[i].candidates for i in idx])
        return [self.model.token_logprobs(items[i].prompt, items[i].completion, self.cfg.temperature, candidates=items[i].candidates) for i in idx]

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

    # ---- data parallelism hooks (single process: every partition runs here; ``DistributedTorchTrainer``: one each) ----
    def _local_parts(self, parts: list[list[int]]) -> list[list[int]]:
        return parts

    def _reduce_grads(self) -> None:
        """Sum gradients across DP ranks (each rank's loss is already its share of the global token mean)."""

    def _reduce_scalars(self, vals: list[float]) -> list[float]:
        return vals

    def step(self, records: Sequence[RolloutRecord], groups: Sequence[str]) -> dict[str, float]:
        items = self._items(records, groups)
        if not items:
            return {"train/items": 0.0}
        lengths = [len(it.completion) for it in items]  # trained tokens: the loss normalizer
        seq_lens = [len(it.prompt) + len(it.completion) for it in items]  # forward tokens: memory and compute, for placement
        g_tokens = sum(lengths)
        single = self.cfg.loss_agg != "token_mean_global"
        parts = balance_partitions(seq_lens, self.cfg.dp) if not single else [list(range(len(items)))]
        mine = sorted(i for p in self._local_parts(parts) for i in p)
        old = None
        src = self.cfg.correction.old_logprob_source
        if self.cfg.correction.name in NEEDS_OLD and src is None:  # checklist #34: implicit recompute only for single-version rollouts
            multi = [r.rollout_id for r in records if not r.mask_sample and len({v for t in r.turns for v in (t.versions() or [t.policy_version])}) > 1]
            if multi:
                raise ValueError(f"correction {self.cfg.correction.name!r} needs π_θold but rollouts {multi[:3]} span several policy versions; "
                                 "set old_logprob_source='recompute_current' (recompute with the current weights) or 'rollout'")
        if self.cfg.correction.name in NEEDS_OLD and src in (None, "recompute_current"):
            with torch.no_grad():  # π_θold for this rank's items only, with the pre-update weights
                old = {i: r.double() for i, r in zip(mine, self._logprobs(items, mine), strict=True)} if mine else {}
        metrics: dict[str, float] = {}
        for _ in range(self.cfg.updates_per_batch):
            self.opt.zero_grad()
            total_loss = 0.0
            mbs = [mb.indices for p in self._local_parts(parts) for mb in micro_batches(seq_lens, p, self.cfg.token_budget)] if not single else [mine]
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
            if not mbs:  # an empty DP rank still joins the collective with zero gradients
                for p in self.model.parameters():
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
            self._reduce_grads()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            self.updates += 1
            keys = sorted({k for m in agg_metrics for k in m})
            sums = self._reduce_scalars([total_loss, float(len(mbs))] + [sum(m.get(k, 0.0) for m in agg_metrics) for k in keys])
            n_mb = sums[1]
            metrics = {"train/loss": sums[0], "train/items": float(len(items)), "train/tokens": float(g_tokens), "train/forward_tokens": float(sum(seq_lens)),
                       "train/micro_batches": n_mb, "train/dp_imbalance": _imbalance(seq_lens, parts)}
            for k, v in zip(keys, sums[2:], strict=True):
                metrics[f"mismatch/{k}"] = v / max(n_mb, 1.0)
        return metrics


def _imbalance(seq_lens: list[int], parts: list[list[int]]) -> float:
    loads = [sum(seq_lens[i] for i in p) for p in parts]
    return max(loads) / (sum(loads) / len(loads)) if loads and sum(loads) else 1.0


class DistributedTorchTrainer(TorchTrainer):
    """Real data parallelism over a ``torch.distributed`` group: every rank holds the full model (DDP-style replicas)
    and the same global batch (``broadcast_batch``: rank ``src`` sends it), computes the same advantages, then trains
    only its token-balanced partition in token-budget micro batches; each micro batch's loss is its share of the
    **global** token mean, so after ``all_reduce(SUM)`` of the gradients every rank holds exactly the full-batch
    gradient and takes the identical optimizer step. ``cfg.dp`` must equal the group size."""

    def __init__(self, model: torch.nn.Module, cfg: TrainerConfig | None = None, *, group: Any = None, broadcast_batch: bool = False, src: int = 0):
        import torch.distributed as dist

        super().__init__(model, cfg)
        self.dist, self.group, self.broadcast_batch, self.src = dist, group, broadcast_batch, src
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        if self.cfg.dp != self.world:
            raise ValueError(f"TrainerConfig.dp={self.cfg.dp} but the group has {self.world} ranks")
        if self.cfg.loss_agg != "token_mean_global":  # only a global-token-mean loss decomposes into per-rank shares that sum
            raise ValueError(f"DistributedTorchTrainer needs loss_agg='token_mean_global' (got {self.cfg.loss_agg!r}): other aggregations do not split across ranks")
        with torch.no_grad():  # replicas start identical
            for p in self.model.parameters():
                dist.broadcast(p.data, src=self.src, group=group)

    def _local_parts(self, parts: list[list[int]]) -> list[list[int]]:
        return [parts[self.rank]] if len(parts) == self.world else parts

    def _reduce_grads(self) -> None:
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        flat = _flatten_dense_tensors(grads)  # one bucket: one collective per update
        self.dist.all_reduce(flat, op=self.dist.ReduceOp.SUM, group=self.group)
        for g, r in zip(grads, _unflatten_dense_tensors(flat, grads), strict=True):
            g.copy_(r)

    def _reduce_scalars(self, vals: list[float]) -> list[float]:
        t = torch.tensor(vals, dtype=torch.float64)
        self.dist.all_reduce(t, op=self.dist.ReduceOp.SUM, group=self.group)
        return t.tolist()

    def step(self, records: Sequence[RolloutRecord], groups: Sequence[str]) -> dict[str, float]:
        if self.broadcast_batch:
            box = [list(records), list(groups)] if self.rank == self.src else [None, None]
            self.dist.broadcast_object_list(box, src=self.src, group=self.group)
            records, groups = box
        return super().step(records, groups)
