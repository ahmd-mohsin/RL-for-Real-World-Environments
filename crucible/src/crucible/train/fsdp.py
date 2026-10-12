"""Sharded data parallelism with PyTorch FSDP2 (M12-infra C; art. §2 "training backends": the FSDP/Megatron side of
every disaggregated stack). ``FSDPTrainer`` is ``DistributedTorchTrainer`` with the parameters, gradients and AdamW
moments sharded over a device mesh (``torch.distributed.fsdp.fully_shard``): a rank holds 1/N of each at rest, so the
model that fits is N times larger than with replicas. A 2D mesh ``(replicate, shard)`` is HSDP: shards inside a node,
replicas across nodes.

The loss is unchanged: each micro batch contributes its share of the **global** token mean, and the reduce-scatter
sums (``set_gradient_divide_factor(1.0)``), so one update equals the single-process full-batch update (the test checks
it to float tolerance against ``TorchTrainer``).

Two resharding modes, because FSDP's collectives run per forward and every rank must issue the same sequence:

- ``reshard="step"`` (default; any model, any forward path): one all-gather per update, forwards and backwards run on
  the gathered weights with local gradient accumulation, and one reduce-scatter at the end (a zero-weighted probe
  backward with gradient sync on carries the accumulated gradients into it). Peak memory holds the full parameters and
  gradients during the update; between updates (rollout time in a colocated topology) only the shards remain.
- ``reshard="layer"`` (ZeRO-3 peak memory): each unit (``units``, e.g. the transformer blocks) gathers in its forward
  and frees after it; gradients reduce-scatter per micro batch. Ranks pad to the same micro-batch count with zero-loss
  probe micro batches, and the π_θold recompute is chunked the same way. Needs one forward per micro batch
  (``batch_token_logprobs``), so per-item paths are refused: ``routing_replay="r3"``, ``prefix_sharing``,
  ``opd_mode="full_vocab"``.

``state_dict()`` is a collective (every rank calls it); the full HF-named state lands on rank 0 (CPU), which publishes
it through the ``WeightSync`` paths unchanged. ``transformer_blocks(model)`` finds the usual unit list of an HF model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from crucible.train.batching import micro_batches
from crucible.train.trainer import DistributedTorchTrainer, TrainerConfig, _Item, _loss_dtype

_METHODS = ("token_logprobs", "batch_token_logprobs", "shared_prefix_logprobs", "completion_logprob_rows")


def transformer_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    """The largest ``ModuleList`` named ``layers`` / ``h`` / ``blocks`` (Llama, Qwen, Mistral: ``model.layers``; GPT-2: ``h``)."""
    best: list[torch.nn.Module] = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and name.rsplit(".", 1)[-1] in ("layers", "h", "blocks") and len(m) > len(best):
            best = list(m)
    return best


def _default_probe(model: torch.nn.Module) -> torch.Tensor:
    if hasattr(model, "batch_token_logprobs"):
        return model.batch_token_logprobs([([0, 0], [0])])[0].sum()
    return model.token_logprobs([0, 0], [0]).sum()


class FSDPTrainer(DistributedTorchTrainer):
    sharded = True  # the state plane writes this trainer's checkpoint with ``save_sharded`` (``train.state``)

    def __init__(self, model: torch.nn.Module, cfg: TrainerConfig | None = None, *, mesh: Any = None, group: Any = None,
                 units: list[torch.nn.Module] | Callable[[torch.nn.Module], list[torch.nn.Module]] | None = None,
                 reshard: str = "step", offload_policy: Any = None, probe: Callable[[torch.nn.Module], torch.Tensor] = _default_probe,
                 broadcast_batch: bool = False, src: int = 0):
        """``mesh``: a ``DeviceMesh`` (1D: FSDP; 2D ``(replicate, shard)``: HSDP); default 1D over ``group`` (or the world)
        on the parameters' device type. ``units``: submodules sharded as their own FSDP units (bottom up), or a function
        of the model; the root is always a unit. ``offload_policy``: FSDP2's ``CPUOffloadPolicy()`` keeps shards and
        optimizer state in host memory."""
        from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
        from torch.distributed.fsdp import FSDPModule, fully_shard, register_fsdp_forward_method

        if reshard not in ("step", "layer"):
            raise ValueError(f"reshard: step | layer (got {reshard!r})")
        super().__init__(model, cfg, group=group, broadcast_batch=broadcast_batch, src=src)  # checks dp == world, token-mean loss; equal init
        c = self.cfg
        if reshard == "layer":
            bad = [k for k, on in (("routing_replay='r3'", c.routing_replay == "r3"), ("prefix_sharing", c.prefix_sharing),
                                   ("opd_mode='full_vocab'", bool(c.opd_coef) and c.opd_mode == "full_vocab")) if on]
            if bad or not hasattr(model, "batch_token_logprobs"):
                raise ValueError(f"reshard='layer' needs one forward per micro batch (batch_token_logprobs) and none of {bad or 'r3/prefix_sharing/full_vocab'}")
        dev = next(model.parameters()).device.type
        if mesh is None:
            mesh = DeviceMesh.from_group(group, dev) if group is not None else init_device_mesh(dev, (self.world,))
        self.mesh, self.reshard, self.probe = mesh, reshard, probe
        kw: dict[str, Any] = {"mesh": mesh}
        if offload_policy is not None:
            kw["offload_policy"] = offload_policy
        blocks = units(model) if callable(units) else list(units or [])
        for u in blocks:
            fully_shard(u, reshard_after_forward=reshard == "layer", **kw)
        fully_shard(model, reshard_after_forward=False, **kw)
        for name in _METHODS:  # the trainer calls these, not forward(): FSDP's root hooks must wrap them
            if hasattr(model, name):
                register_fsdp_forward_method(model, name)
        self._fsdp = [m for m in model.modules() if isinstance(m, FSDPModule)]
        for m in self._fsdp:
            m.set_gradient_divide_factor(1.0)  # sum: each rank's loss is already its share of the global token mean
            if self.dist.get_backend(group) != "nccl":
                m.set_force_sum_reduction_for_comms(True)  # gloo has no PREMUL_SUM: plain SUM collectives
            if reshard == "step":
                m.set_requires_gradient_sync(False, recurse=False)
                m.set_reshard_after_backward(False, recurse=False)
        self.reset_optimizer()  # AdamW over the sharded (DTensor) parameters: the moments are sharded too
        self.counters = {"all_gathers": 0, "probe_micro_batches": 0}

    # ---- resharding hooks ---------------------------------------------------------------------------------------
    def _gather_params(self) -> None:
        if self.reshard == "step" and not getattr(self, "_gathered", False):
            for m in self._fsdp:
                m.unshard()
            self._gathered = True
            self.counters["all_gathers"] += 1

    def _release_params(self) -> None:
        if self.reshard == "step" and getattr(self, "_gathered", False):
            for m in self._fsdp:
                m.reshard()
            self._gathered = False

    def _reduce_grads(self) -> None:
        if self.reshard == "layer":  # already reduce-scattered, per micro batch
            return
        for m in self._fsdp:
            m.set_requires_gradient_sync(True, recurse=False)
        (0.0 * self.probe(self.model)).backward()  # carries the locally accumulated gradients into one reduce-scatter
        for m in self._fsdp:
            m.set_requires_gradient_sync(False, recurse=False)
        self._release_params()  # the module now holds the sharded parameters, whose .grad the clip and optimizer see
        self.counters["probe_micro_batches"] += 1

    # ---- lockstep micro batches (layer mode) ----------------------------------------------------------------------
    def _max_count(self, n: int) -> int:
        t = torch.tensor([n], dtype=torch.int64)
        self.dist.all_reduce(t, op=self.dist.ReduceOp.MAX, group=self.group)
        return int(t.item())

    def _equalize(self, mbs: list[list[int]]) -> list[list[int]]:
        if self.reshard != "layer":
            return mbs
        return mbs + [[]] * (self._max_count(len(mbs)) - len(mbs))

    def _idle_micro_batch(self) -> None:
        (0.0 * self.probe(self.model)).backward()
        self.counters["probe_micro_batches"] += 1

    def _old_logprobs(self, items: list[_Item], mine: list[int], seq_lens: list[int], parts: list[list[int]]) -> dict[int, torch.Tensor]:
        if self.reshard != "layer":
            return super()._old_logprobs(items, mine, seq_lens, parts)
        chunks = self._equalize([mb.indices for p in self._local_parts(parts) for mb in micro_batches(seq_lens, p, self.cfg.token_budget)])
        out: dict[int, torch.Tensor] = {}
        for idx in chunks:
            if not idx:
                self.probe(self.model)
                continue
            for i, r in zip(idx, self._logprobs(items, idx), strict=True):
                out[i] = r.to(_loss_dtype(r.device))
        return out

    # ---- publishing and memory ----------------------------------------------------------------------------------
    def state_dict(self) -> dict[str, torch.Tensor]:
        """Collective: every rank must call it. The full state on rank 0 (CPU); ``{}`` on the others."""
        self._release_params()
        out: dict[str, torch.Tensor] = {}
        for k, v in self.model.state_dict().items():  # the module's own names (HF names for HFPolicy), in the same order on every rank
            full = v.full_tensor() if hasattr(v, "full_tensor") else v  # one all-gather per sharded tensor
            if self.rank == 0:
                out[k] = full.detach().to("cpu", copy=True)
        return out

    def _flat_state(self, *, init_moments: bool) -> dict[str, Any]:
        """Parameters and AdamW state keyed by parameter name, as the (sharded) DTensors themselves: DCP writes and
        reads each rank's shards in place, with no gather to one rank."""
        self._release_params()
        out: dict[str, Any] = {}
        for name, prm in self.model.named_parameters():
            out[f"model/{name}"] = prm
            st = self.opt.state[prm]
            if init_moments and not st:  # a fresh optimizer: shaped targets for the load
                st.update(step=torch.zeros((), dtype=torch.float32), exp_avg=torch.zeros_like(prm), exp_avg_sq=torch.zeros_like(prm))
            for k, v in st.items():
                out[f"optim/{name}/{k}"] = v
        out["updates"] = torch.tensor(self.updates)
        return out

    def save_sharded(self, path: str) -> None:
        """Collective: every rank writes its own shards (``torch.distributed.checkpoint``); resharding on load is allowed
        (a different world size reads the same checkpoint)."""
        import torch.distributed.checkpoint as dcp

        dcp.save(self._flat_state(init_moments=False), checkpoint_id=path)

    def load_sharded(self, path: str) -> None:
        import torch.distributed.checkpoint as dcp

        st = self._flat_state(init_moments=True)
        dcp.load(st, checkpoint_id=path)  # in place into the parameter and moment shards
        self.updates = int(st["updates"])

    def local_numel(self) -> dict[str, int]:
        """Elements this rank holds at rest: parameter shards and AdamW moment shards (the memory claim, measured)."""
        par = sum(p.to_local().numel() if hasattr(p, "to_local") else p.numel() for p in self.model.parameters())
        mom = sum((v.to_local().numel() if hasattr(v, "to_local") else v.numel()) for st in self.opt.state.values()
                  for k, v in st.items() if torch.is_tensor(v) and k != "step")
        return {"params": par, "moments": mom}

    def offload(self, mode: str = "cpu", *, path: str | None = None) -> None:
        if mode != "none":
            raise NotImplementedError("FSDPTrainer: pass offload_policy=CPUOffloadPolicy() (shards and moments live in host memory)")

    def onload(self) -> None:
        return None
