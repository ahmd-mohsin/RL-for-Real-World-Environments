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
    routing_replay: str = "none"  # none | r3: replay the engine's MoE expert choice per completion token (checklist #36)
    prefix_sharing: bool = False  # items with the same full prompt in a micro batch encode it once (checklist #50)
    # "segment" (default): one causal sequence per (rollout, actor, compaction segment), loss only on sampled tokens:
    # forward tokens O(turns) instead of O(turns²) (m13 B-1). "call": one item per model call with its full prompt.
    item_unit: str = "segment"


@dataclass
class _Item:
    prompt: list[int]
    completion: list[int]
    rollout_lp: list[float]
    adv: float | list[float]  # one value broadcast over the call's tokens, or per token (critic GAE)
    prompt_index: int
    candidates: list[list[int]] | None = None  # top-p nucleus per token: π_θ is renormalized over it (checklist #37)
    routing: list[list[int]] | None = None  # MoE experts per token, replayed under routing_replay="r3"
    mask: list[int] | None = None  # per completion position: 1 = trained (sampled), 0 = observation (segment items); None = all


class TorchTrainer:
    def __init__(self, model: torch.nn.Module, cfg: TrainerConfig | None = None, *, critic: torch.nn.Module | None = None, critic_cfg=None):
        """``critic``: a value model (``critic.TinyValue`` / ``HFValue``): critic mode — token-level GAE advantages instead of
        group-relative ones (works with n = 1), and ``critic_cfg.updates_per_policy_update`` critic steps per policy step."""
        self.model, self.cfg = model, cfg or TrainerConfig()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        self.updates = 0
        self.critic = critic
        if critic is not None:
            from crucible.train.critic import CriticConfig

            self.critic_cfg = critic_cfg or CriticConfig()
            self.critic_opt = torch.optim.AdamW([q for q in critic.parameters() if q.requires_grad], lr=self.critic_cfg.lr)
            self.critic_updates = 0

    def reset_optimizer(self) -> None:
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)

    # ---- offload (checklist #7): free the accelerator for the engine while rollouts run ---------------------------
    def offload(self, mode: str = "cpu", *, path: str | None = None) -> None:
        """``cpu``: parameters and optimizer moments to host memory, gradients dropped (they are zeroed at the next step
        anyway). ``nvme``: parameters and moments written to ``path`` and released (host RAM too). ``onload()``
        restores them, bit for bit, to the device they came from."""
        if mode == "none" or getattr(self, "_offloaded", None):
            return
        if mode not in ("cpu", "nvme"):
            raise ValueError("offload mode: none | cpu | nvme")
        params = list(self.model.parameters())
        self._home = params[0].device
        for prm in params:
            prm.grad = None
        state = self.opt.state
        if mode == "cpu":
            for prm in params:
                prm.data = prm.data.to("cpu")
            for st in state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to("cpu")
        else:
            import tempfile

            self._nvme = path or tempfile.mkstemp(prefix="crucible-offload-", suffix=".pt")[1]
            torch.save({"params": [prm.data.to("cpu") for prm in params],
                        "opt": {i: {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in state[prm].items()} for i, prm in enumerate(params) if prm in state}},
                       self._nvme)
            for prm in params:
                prm.data = torch.empty(0, dtype=prm.dtype)
            for prm in params:
                if prm in state:
                    state[prm] = {}
        self._offloaded = mode
        import gc

        gc.collect()  # the last step's graph leaves reference cycles that pin device tensors (measured: a parameter-sized residue)
        if self._home.type == "cuda":
            if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):  # cuBLAS workspaces live in the caching allocator (64 MiB on the lab 5090)
                torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
        elif self._home.type == "mps":
            torch.mps.empty_cache()

    def onload(self) -> None:
        mode = getattr(self, "_offloaded", None)
        if not mode:
            return
        params = list(self.model.parameters())
        if mode == "cpu":
            for prm in params:
                prm.data = prm.data.to(self._home)
            for st in self.opt.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v) and k != "step":
                        st[k] = v.to(self._home)
        else:
            blob = torch.load(self._nvme, weights_only=True)
            for prm, data in zip(params, blob["params"], strict=True):
                prm.data = data.to(self._home)
            for i, st in blob["opt"].items():
                self.opt.state[params[int(i)]] = {k: (v.to(self._home) if torch.is_tensor(v) and k != "step" else v) for k, v in st.items()}
            import os

            os.unlink(self._nvme)
        self._offloaded = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def _gae_advantages(self, records: Sequence[RolloutRecord]) -> dict[tuple[str, str], list[float]]:
        """Critic mode: per (rollout id, call id), the per-token GAE advantages (call ids repeat across rollouts: World
        episodes all have ``agent:c1``...); also fills ``self._critic_data`` with (prompt, completion,
        returns) for the critic regression. The chain holds only completion tokens (skip-observation), in call order."""
        from crucible.train.critic import explained_variance
        from crucible.train.objectives import gae, length_adaptive_lambda

        cc = self.critic_cfg
        out: dict[tuple[str, str], list[float]] = {}
        self._critic_data: list[tuple[list[int], list[int], torch.Tensor]] = []
        all_ret, all_val = [], []
        for r in records:
            if r.mask_sample or r.reward is None:
                continue
            prompts = full_prompts(r)
            calls = [t for t in r.turns if t.completion_token_ids]
            if not calls:
                continue
            with torch.no_grad():
                vals = [self.critic.token_values(prompts[t.call_id], list(t.completion_token_ids)).detach().double().cpu() for t in calls]
            lens = [len(t.completion_token_ids) for t in calls]
            v = torch.cat(vals)
            T = int(v.shape[0])
            rewards = torch.zeros(T, dtype=torch.float64)
            rewards[-1] = float(r.reward)
            seg = torch.tensor([t.segment for t, n in zip(calls, lens, strict=True) for _ in range(n)])
            multi = len(set(seg.tolist())) > 1
            kw = {"segments": seg, "shared_terminal": float(r.reward)} if multi else {}
            if multi:
                rewards[-1] = 0.0  # the shared terminal carries the outcome to every segment's end
            lam_p = length_adaptive_lambda(T, cc.alpha) if cc.lam_policy == "adaptive" else float(cc.lam_policy)
            trunc = {"truncated": bool(r.truncated), "bootstrap_value": float(v[-1])}
            adv, _ = gae(rewards, v, torch.ones(T, dtype=torch.bool), gamma=cc.gamma, lam=lam_p, **kw, **trunc)
            _, ret = gae(rewards, v, torch.ones(T, dtype=torch.bool), gamma=cc.gamma, lam=cc.lam_critic, **kw, **trunc)
            k = 0
            for t, n in zip(calls, lens, strict=True):
                out[(r.rollout_id, t.call_id)] = adv[k:k + n].tolist()
                self._critic_data.append((prompts[t.call_id], list(t.completion_token_ids), ret[k:k + n].float()))
                k += n
            all_ret.append(ret.float())
            all_val.append(v.float())
        self._ev = explained_variance(torch.cat(all_ret), torch.cat(all_val)) if all_ret else 0.0
        return out

    def _critic_step(self) -> dict[str, float]:
        """TTUR: K critic regressions (MSE to the λ_critic returns, global token mean) per policy update."""
        cc, data = self.critic_cfg, self._critic_data
        if not data:
            return {}
        n_tok = sum(len(c) for _, c, _ in data)
        loss_v = 0.0
        for _ in range(cc.updates_per_policy_update):
            self.critic_opt.zero_grad()
            total = 0.0
            for prompt, comp, ret in data:
                v = self.critic.token_values(prompt, comp)
                loss = ((v - ret.to(v.device)) ** 2).sum() / n_tok
                loss.backward()
                total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_([q for q in self.critic.parameters() if q.requires_grad], cc.grad_clip)
            self.critic_opt.step()
            self.critic_updates += 1
            loss_v = total
        return {"critic/loss": loss_v, "critic/explained_variance": self._ev, "critic/updates": float(self.critic_updates)}

    def _segment_ok(self, r: RolloutRecord) -> bool:
        """Segment packing needs no per-call model inputs: truncated sampling (candidates) and R3 routing keep per-call items."""
        if self.cfg.routing_replay == "r3":
            return False
        return all(float(t.sampling.get("top_p", 1.0)) >= 1.0 for t in r.turns)

    def _segment_items(self, r: RolloutRecord, a: float, per_token: dict | None, pid: int) -> list[_Item]:
        from crucible.train.bridges.common import flatten

        out = []
        for seq in flatten(r, seq_adv=a, turn_weight=self.cfg.turn_weight):
            adv = list(seq.advantages)
            if per_token is not None:  # critic GAE: per-token advantages on each call's own span (+ turn credit)
                credit = {t.call_id: self.cfg.turn_weight * (t.advantage or 0.0) for t in r.turns}
                for cid, s0, s1 in seq.call_spans:
                    vals = per_token.get((r.rollout_id, cid))
                    if vals is not None:
                        adv[s0:s1] = [x + credit.get(cid, 0.0) for x in vals[: s1 - s0]]
            if sum(seq.response_mask):
                out.append(_Item(list(seq.prompt_ids), list(seq.response_ids), list(seq.logprobs), adv, pid, mask=list(seq.response_mask)))
        return out

    def _items(self, records: Sequence[RolloutRecord], groups: Sequence[str]) -> list[_Item]:
        if self.critic is not None:
            per_token = self._gae_advantages(records)
            advs = [0.0] * len(records)
        else:
            per_token = None
            advs = group_advantages([r.reward if not r.mask_sample else None for r in records], groups, std_normalize=self.cfg.std_normalize,
                                estimator=self.cfg.estimator)
            # a group stage (m13 C-2) that set every scored member's advantage replaces the group statistic for that group
            scored: dict[str, list[int]] = {}
            for i, (r, g) in enumerate(zip(records, groups, strict=True)):
                if not r.mask_sample and r.reward is not None:
                    scored.setdefault(g, []).append(i)
            for idx in scored.values():
                if all(records[i].advantage is not None for i in idx):
                    for i in idx:
                        advs[i] = float(records[i].advantage)
        pids = {g: i for i, g in enumerate(dict.fromkeys(groups))}
        items = []
        for r, a, g in zip(records, advs, groups, strict=True):
            if r.mask_sample or r.reward is None:
                continue
            if self.cfg.item_unit == "segment" and self._segment_ok(r):
                items += self._segment_items(r, a, per_token, pids[g])
                continue
            prompts = full_prompts(r)
            for t in r.turns:
                if not t.completion_token_ids:
                    continue
                if not t.candidates_valid():
                    raise ValueError(f"{t.call_id}: sampled with top_p={t.sampling.get('top_p')} but no complete candidate sets; the trainer cannot "
                                     "renormalize π_θ (record candidates or sample with top_p=1)")
                if self.cfg.routing_replay == "r3" and (t.routing is None or len(t.routing) != len(t.completion_token_ids)):
                    raise ValueError(f"{t.call_id}: routing_replay='r3' needs Turn.routing for 100% of completion tokens "
                                     f"({0 if t.routing is None else len(t.routing)}/{len(t.completion_token_ids)})")
                credit = self.cfg.turn_weight * (t.advantage or 0.0)
                adv = [x + credit for x in per_token[(r.rollout_id, t.call_id)]] if per_token is not None else a + credit
                items.append(_Item(prompts[t.call_id], list(t.completion_token_ids), list(t.logprobs or []), adv, pids[g],
                                   t.candidates if float(t.sampling.get("top_p", 1.0)) < 1.0 else None,
                                   t.routing if self.cfg.routing_replay == "r3" else None))
        return items

    def _logprobs(self, items: list[_Item], idx: list[int]) -> list[torch.Tensor]:
        if self.cfg.routing_replay == "r3":  # per item: the replayed expert path
            out = []
            for i in idx:
                out.append(self.model.token_logprobs(items[i].prompt, items[i].completion, self.cfg.temperature, candidates=items[i].candidates,
                                                     routing=items[i].routing))
                self._confidence.append(float(getattr(self.model, "last_confidence", 1.0)))
            return out
        share = getattr(self.model, "shared_prefix_logprobs", None) if self.cfg.prefix_sharing else None
        if share is not None:  # group by identical full prompt; truncated-sampling items keep the plain path
            by: dict[tuple, list[int]] = {}
            for i in idx:
                if items[i].candidates is None:
                    by.setdefault(tuple(items[i].prompt), []).append(i)
            res: dict[int, torch.Tensor] = {}
            for prompt, members in by.items():
                if len(members) > 1:
                    for i, lp in zip(members, share(list(prompt), [items[j].completion for j in members], self.cfg.temperature), strict=True):
                        res[i] = lp
                    self._saved_tokens += (len(members) - 1) * len(prompt)
            rest = [i for i in idx if i not in res]
            if rest:
                plain = TorchTrainer._plain_logprobs(self, items, rest)
                res.update(zip(rest, plain, strict=True))
            return [res[i] for i in idx]
        return self._plain_logprobs(items, idx)

    def _plain_logprobs(self, items: list[_Item], idx: list[int]) -> list[torch.Tensor]:
        if hasattr(self.model, "batch_token_logprobs"):  # one forward per micro batch (HF policies)
            return self.model.batch_token_logprobs([(items[i].prompt, items[i].completion) for i in idx], self.cfg.temperature,
                                                   candidates=[items[i].candidates for i in idx])
        out = [self.model.token_logprobs(items[i].prompt, items[i].completion, self.cfg.temperature, candidates=items[i].candidates) for i in idx]
        if hasattr(self.model, "last_confidence"):
            self._confidence.append(float(self.model.last_confidence))
        return out

    def _padded(self, items: list[_Item], idx: list[int], old: dict[int, torch.Tensor] | None):
        T = max(len(items[i].completion) for i in idx)
        rows = self._logprobs(items, idx)
        dev = rows[0].device
        fdt = _loss_dtype(dev)
        roll = torch.zeros(len(idx), T, dtype=fdt, device=dev)
        adv = torch.zeros(len(idx), T, dtype=fdt, device=dev)
        mask = torch.zeros(len(idx), T, device=dev)
        old_t = torch.zeros(len(idx), T, dtype=fdt, device=dev)
        lp_rows = []
        for k, i in enumerate(idx):
            it = items[i]
            n = len(it.completion)
            lp_rows.append(torch.nn.functional.pad(rows[k].to(fdt), (0, T - n)))
            roll[k, :n] = torch.tensor(it.rollout_lp[:n], dtype=fdt, device=dev)
            adv[k, :n] = torch.tensor(it.adv[:n], dtype=fdt, device=dev) if isinstance(it.adv, list) else it.adv
            mask[k, :n] = torch.tensor(it.mask[:n], dtype=mask.dtype, device=dev) if it.mask is not None else 1
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
        if self.cfg.routing_replay not in ("none", "r3"):
            raise ValueError(f"routing_replay: none | r3 (got {self.cfg.routing_replay!r})")
        self._confidence: list[float] = []
        self._saved_tokens = 0
        items = self._items(records, groups)
        if not items:
            return {"train/items": 0.0}
        lengths = [sum(it.mask) if it.mask is not None else len(it.completion) for it in items]  # trained tokens: the loss normalizer
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
                old = {i: r.to(_loss_dtype(r.device)) for i, r in zip(mine, self._logprobs(items, mine), strict=True)} if mine else {}
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
            if self.cfg.prefix_sharing:
                metrics["train/prefix_tokens_saved"] = float(self._saved_tokens)
            if self._confidence:  # MoE routing confidence Σ_k (mean top-k router mass), checklist #36
                metrics["moe/routing_confidence"] = sum(self._confidence) / len(self._confidence)
        if self.critic is not None:
            metrics.update(self._critic_step())
        return metrics


def _loss_dtype(dev: torch.device) -> torch.dtype:
    """Per-token loss terms in float64 (ratios of near-equal probabilities); MPS has no float64, so float32 there."""
    return torch.float32 if dev.type == "mps" else torch.float64


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
