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

Scale-out (M12 audit H4 V2/V4/V7/V14/V17, G5 D2/D3/P2-a):

- **Loading without a full copy per GPU.** Build the model under ``meta_params()`` on every rank (parameters on the
  ``meta`` device, buffers real) and pass the full state on the source rank only (``full_state=``, e.g. rank 0 reads
  the checkpoint into host memory): after ``fully_shard`` each rank allocates its shards only and receives them tensor
  by tensor (the pattern of torch's ``set_model_state_dict(full_state_dict=True, broadcast_from_rank0=True)``, as verl
  and NeMo-RL load). Peak GPU memory is the shards plus one full tensor, not ``4·params``. The replica broadcast of the
  parent constructor is skipped (``sync_init=False``).
- **A parameter some rank never reaches** (an idle MoE expert, a vision tower on a text-only rank, a conditional head):
  ``step`` mode agrees on the set of parameters with a gradient on any rank and zero-fills the unsharded gradients of the
  others before the one reduce-scatter (a parameter no rank reached keeps no gradient, as in ``TorchTrainer``). ``layer``
  mode reduce-scatters a zero for an unreached parameter of a unit that ran (FSDP2 ``set_reduce_scatter_unused_params``;
  AdamW then decays it as if its gradient were 0); a unit whose forward ran on some ranks only cannot be gathered in
  lockstep, so every rank checks the unit order before each gather and raises a clear error instead of hanging
  (``unit_check``).
- **Offload** (``offload("cpu")``): shards and AdamW moments move to host memory between steps and the CUDA cache is
  emptied, so a colocated engine gets the GPU (verl's FSDP2 offload pattern, ``model.cpu()`` + optimizer state).
- **Mixed precision** (``mixed_precision="bf16"``): the units gather bf16 parameters and reduce-scatter fp32 gradients
  (FSDP2 ``MixedPrecisionPolicy``, verl's default); the root unit (embeddings, final norm, LM head) stays fp32, so the
  fp32 LM head of ``HFPolicy`` keeps fp32 weights.
- ``mesh_shape=(replicate, shard)`` builds the HSDP mesh the planner chose (``Placement.trainer_kwargs()``).
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterator
from typing import Any

import torch

from crucible.train.trainer import DistributedTorchTrainer, TrainerConfig, _Item, _loss_dtype

_METHODS = ("token_logprobs", "batch_token_logprobs", "shared_prefix_logprobs", "completion_logprob_rows")


def transformer_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    """The largest ``ModuleList`` named ``layers`` / ``h`` / ``blocks`` (Llama, Qwen, Mistral: ``model.layers``; GPT-2: ``h``)."""
    best: list[torch.nn.Module] = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and name.rsplit(".", 1)[-1] in ("layers", "h", "blocks") and len(m) > len(best):
            best = list(m)
    return best


@contextlib.contextmanager
def meta_params() -> Iterator[None]:
    """Build a model with its **parameters** on the ``meta`` device (no memory) and its buffers real (rotary tables and
    other non-persistent buffers are computed as usual): the pattern of accelerate's ``init_empty_weights(include_buffers=False)``.
    Pass the result to ``FSDPTrainer`` with ``full_state=`` on the source rank."""
    orig = torch.nn.Module.register_parameter

    def register(self: torch.nn.Module, name: str, param: torch.nn.Parameter | None) -> None:
        orig(self, name, param)
        if param is not None and not param.is_meta:
            p = self._parameters[name]
            self._parameters[name] = type(p)(p.to("meta"), requires_grad=p.requires_grad)

    torch.nn.Module.register_parameter = register  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = orig  # type: ignore[method-assign]


def _lookup(full: dict[str, torch.Tensor], name: str) -> torch.Tensor | None:
    """A full-state tensor under the module's parameter name, or under that name with a wrapper prefix stripped
    (``HFPolicy`` names its state like the HF model: ``model.lm_head.weight`` → ``lm_head.weight``)."""
    if name in full:
        return full[name]
    for pre in ("model.", "module.", "_orig_mod."):
        if name.startswith(pre) and name[len(pre):] in full:
            return full[name[len(pre):]]
    return None


def _fsdp_params(m: Any) -> list[Any]:
    st = m._get_fsdp_state()
    groups = getattr(st, "_fsdp_param_groups", None)
    if groups is None:  # torch < 2.10: one group per state
        groups = [st._fsdp_param_group] if st._fsdp_param_group is not None else []
    return [fp for g in groups for fp in g.fsdp_params]


def _default_probe(model: torch.nn.Module) -> torch.Tensor:
    if hasattr(model, "batch_token_logprobs"):
        return model.batch_token_logprobs([([0, 0], [0])])[0].sum()
    return model.token_logprobs([0, 0], [0]).sum()


class FSDPTrainer(DistributedTorchTrainer):
    sharded = True  # the state plane writes this trainer's checkpoint with ``save_sharded`` (``train.state``)

    def __init__(self, model: torch.nn.Module, cfg: TrainerConfig | None = None, *, mesh: Any = None, group: Any = None,
                 units: list[torch.nn.Module] | Callable[[torch.nn.Module], list[torch.nn.Module]] | None = None,
                 reshard: str = "step", offload_policy: Any = None, probe: Callable[[torch.nn.Module], torch.Tensor] = _default_probe,
                 broadcast_batch: bool = False, src: int = 0, mesh_shape: tuple[int, int] | None = None, device: Any = None,
                 full_state: dict[str, torch.Tensor] | None = None, mixed_precision: Any = None, unit_check: bool = True,
                 critic: torch.nn.Module | None = None, critic_cfg: Any = None, teacher: torch.nn.Module | None = None):
        """``mesh``: a ``DeviceMesh`` (1D: FSDP; 2D ``(replicate, shard)``: HSDP); default 1D over ``group`` (or the world)
        on the parameters' device type; ``mesh_shape=(replicate, shard)`` builds the 2D mesh instead (the planner's
        ``Placement.trainer_kwargs()``). ``units``: submodules sharded as their own FSDP units (bottom up), or a function
        of the model; the root is always a unit. ``offload_policy``: FSDP2's ``CPUOffloadPolicy()`` keeps shards and
        optimizer state in host memory.

        ``device``: where the shards live (default: the parameters' device; for a ``meta_params()`` model, the current
        CUDA device on NCCL, else the CPU). ``full_state``: with a meta model, the full state dict on group rank ``src``
        (``None`` elsewhere); it is scattered shard by shard. ``mixed_precision``: ``"bf16"`` / ``"fp16"`` or an FSDP2
        ``MixedPrecisionPolicy`` for the units (the root stays fp32). ``unit_check`` (layer mode): every rank agrees on
        each unit before gathering it, so data-dependent units raise instead of hanging."""
        from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
        from torch.distributed.fsdp import FSDPModule, fully_shard, register_fsdp_forward_method

        if reshard not in ("step", "layer"):
            raise ValueError(f"reshard: step | layer (got {reshard!r})")
        meta = any(p.is_meta for p in model.parameters())
        # a meta model is materialized from ``full_state`` below: no replica broadcast of the full model (H4 V4 / G5 D2)
        super().__init__(model, cfg, group=group, broadcast_batch=broadcast_batch, src=src, critic=critic, critic_cfg=critic_cfg,
                         teacher=teacher, sync_init=not meta)  # checks dp == world
        backend = self.dist.get_backend(group)
        if device is None:
            device = (torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else torch.device("cpu")) if meta \
                else next(model.parameters()).device
        device = torch.device(device)
        if backend == "nccl":
            self._coll_dev = device
        if meta and critic is not None:  # the parent skipped every broadcast: the critic's replicas start equal here
            with torch.no_grad():
                for p in critic.parameters():
                    self.dist.broadcast(p.data, group_src=self.src, group=group)
        c = self.cfg
        if reshard == "layer":
            bad = [k for k, on in (("routing_replay='r3'", c.routing_replay == "r3"), ("prefix_sharing", c.prefix_sharing),
                                   ("opd_mode='full_vocab'", bool(c.opd_coef) and c.opd_mode == "full_vocab")) if on]
            if bad or not hasattr(model, "batch_token_logprobs"):
                raise ValueError(f"reshard='layer' needs one forward per micro batch (batch_token_logprobs) and none of {bad or 'r3/prefix_sharing/full_vocab'}")
        dev = device.type
        if mesh is not None and mesh_shape is not None:
            raise ValueError("pass mesh or mesh_shape, not both")
        if mesh is None and mesh_shape is not None:
            rep, shard = (int(x) for x in mesh_shape)
            if rep * shard != self.world:
                raise ValueError(f"mesh_shape {tuple(mesh_shape)} has {rep * shard} ranks but the group has {self.world}")
            if group is not None and rep > 1:
                raise ValueError("mesh_shape with replicas needs the default group (build the DeviceMesh yourself for a subgroup)")
            mesh = init_device_mesh(dev, (rep, shard), mesh_dim_names=("replicate", "shard")) if rep > 1 else None
        if mesh is None:
            mesh = DeviceMesh.from_group(group, dev) if group is not None else init_device_mesh(dev, (self.world,))
        self.mesh, self.reshard, self.probe, self.device = mesh, reshard, probe, device
        self._host_resident = offload_policy is not None  # FSDP2 CPUOffloadPolicy: shards and moments already live in host memory
        kw: dict[str, Any] = {"mesh": mesh}
        if offload_policy is not None:
            kw["offload_policy"] = offload_policy
        for mod in model.modules():  # cuDNN RNNs re-point their weights into one flat buffer on CUDA (``flatten_parameters``),
            if isinstance(mod, torch.nn.RNNBase):  # detaching them from FSDP's gathered storage: the module would read stale weights
                mod.flatten_parameters = lambda: None  # found on NCCL (Modal L4), invisible on CPU; cuDNN then reads them in place
        blocks = units(model) if callable(units) else list(units or [])
        mp = _mp_policy(mixed_precision)
        if mp is not None and not blocks:
            raise ValueError("mixed_precision applies to the units (the root keeps fp32 for the LM head): pass units=")
        if meta and self.rank == self.src and full_state is None:
            raise ValueError("a meta-device model needs full_state= on the source rank (FSDPTrainer loads the shards from it)")
        bufs = {n: b.detach().clone() for n, b in model.named_buffers() if not b.is_meta} if meta else {}
        for u in blocks:
            fully_shard(u, reshard_after_forward=reshard == "layer", **kw, **({"mp_policy": mp} if mp is not None else {}))
        fully_shard(model, reshard_after_forward=False, **kw)
        if meta:
            self._materialize(full_state or {}, bufs)
        if reshard == "layer" and unit_check:
            self._install_unit_check(model, blocks)
        for name in _METHODS:  # the trainer calls these, not forward(): FSDP's root hooks must wrap them
            if hasattr(model, name):
                register_fsdp_forward_method(model, name)
        self._fsdp = [m for m in model.modules() if isinstance(m, FSDPModule)]
        for m in self._fsdp:
            m.set_gradient_divide_factor(1.0)  # sum: each rank's loss is already its share of the global token mean
            if backend != "nccl":
                m.set_force_sum_reduction_for_comms(True)  # gloo has no PREMUL_SUM: plain SUM collectives
            if reshard == "step":
                m.set_requires_gradient_sync(False, recurse=False)
                m.set_reshard_after_backward(False, recurse=False)
            else:  # an unreached parameter of a unit that ran: a zero joins the reduce-scatter on every rank (H4 V2)
                m.set_reduce_scatter_unused_params(True, recurse=False)
        self.reset_optimizer()  # AdamW over the sharded (DTensor) parameters: the moments are sharded too
        self.counters = {"all_gathers": 0, "probe_micro_batches": 0, "zero_filled_grads": 0}

    # ---- loading from the source rank (meta-device init) ---------------------------------------------------------
    def _materialize(self, full: dict[str, torch.Tensor], bufs: dict[str, torch.Tensor]) -> None:
        """Allocate this rank's shards (``to_empty``) and fill them from the source rank's full state, one tensor at a
        time: broadcast the full tensor over the trainer group, keep the local shard (``distribute_tensor`` with
        ``src_data_rank=None`` slices locally, matching FSDP2's padded chunks), free the full tensor. Buffers: the
        source rank's (restored after ``to_empty``) are broadcast too, so every replica computes with the same ones."""
        from torch.distributed.tensor import DTensor, distribute_tensor

        model, dev = self.model, self.device
        if self.rank == self.src:
            missing = [n for n, _ in model.named_parameters() if _lookup(full, n) is None]
            if missing:
                raise KeyError(f"full_state lacks {len(missing)} parameters, e.g. {missing[:5]}")
        model.to_empty(device=dev)
        with torch.no_grad():
            for n, b in model.named_buffers():
                if n in bufs:
                    b.copy_(bufs[n].to(dev))
            for n, p in model.named_parameters():
                t = _lookup(full, n) if self.rank == self.src else None
                buf = t.detach().to(device=dev, dtype=p.dtype) if t is not None else torch.empty(p.shape, dtype=p.dtype, device=dev)
                if tuple(buf.shape) != tuple(p.shape):
                    raise ValueError(f"full_state[{n!r}] has shape {tuple(buf.shape)}, the model {tuple(p.shape)}")
                self.dist.broadcast(buf, group_src=self.src, group=self.group)
                if isinstance(p, DTensor):
                    local = distribute_tensor(buf, p.device_mesh, p.placements, src_data_rank=None).to_local()
                    p.to_local().copy_(local)
                else:
                    p.copy_(buf)
                del buf
            for _, b in model.named_buffers():
                self.dist.broadcast(b, group_src=self.src, group=self.group)
        self._empty_cache()

    # ---- layer mode: lockstep unit order (H4 V2) -------------------------------------------------------------------
    def _install_unit_check(self, model: torch.nn.Module, blocks: list[torch.nn.Module]) -> None:
        """Before a unit's all-gather, every rank all-reduces the unit's index (and at the end of each root forward a
        sentinel) on a CPU group: a unit that runs on some ranks only (data-dependent control flow) would pair its
        all-gather with another unit's on the other ranks (a hang on NCCL, garbage or an abort on gloo); here every rank
        raises the same error instead."""
        ranks = self.dist.get_process_group_ranks(self.group) if self.group is not None else list(range(self.dist.get_world_size()))
        self._check_group = self.group if self.dist.get_backend(self.group) == "gloo" else \
            self.dist.new_group(ranks=ranks, backend="gloo", use_local_synchronization=True)
        names = {id(m): n for n, m in model.named_modules()}

        def check(uid: int, what: str) -> None:
            t = torch.tensor([uid, -uid], dtype=torch.int64)
            self.dist.all_reduce(t, op=self.dist.ReduceOp.MAX, group=self._check_group)
            if int(t[0]) != -int(t[1]):
                lo, hi = -int(t[1]), int(t[0])
                label = {i: names.get(id(b), str(i)) for i, b in enumerate(blocks)}
                raise RuntimeError(f"FSDPTrainer(reshard='layer'): the ranks ran different units at the same point ({what}: "
                                   f"{label.get(lo, 'end of forward')} vs {label.get(hi, 'end of forward')}); a unit whose forward depends "
                                   "on the data (an MoE expert, a vision tower) cannot be gathered in lockstep: use reshard='step', or "
                                   "make that module part of a unit that always runs")

        for i, b in enumerate(blocks):
            b.register_forward_pre_hook(lambda _m, _a, i=i: check(i, "unit forward"), prepend=True)
        end = len(blocks)
        for name in _METHODS:
            fn = getattr(model, name, None)
            if fn is None:
                continue

            @functools.wraps(fn)  # keeps the signature (``_takes_media`` inspects it)
            def wrapped(*a: Any, _fn: Callable = fn, **k: Any) -> Any:
                out = _fn(*a, **k)
                check(end, "end of forward")
                return out

            setattr(model, name, wrapped)

    # ---- resharding hooks ---------------------------------------------------------------------------------------
    def _gather_params(self) -> None:
        if self.reshard == "step" and not getattr(self, "_gathered", False):
            dev = self._coll_dev
            if dev.type == "cuda":  # precaution: FSDP all-gathers on its own stream, and only the root's pre-forward orders it
                torch.cuda.current_stream(dev).synchronize()  # after the optimizer step; an explicit unshard is ordered here instead
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
        unused = self._zero_fill_unsharded()
        for m in self._fsdp:
            m.set_requires_gradient_sync(True, recurse=False)
        (0.0 * self.probe(self.model)).backward()  # carries the locally accumulated gradients into one reduce-scatter
        for m in self._fsdp:
            m.set_requires_gradient_sync(False, recurse=False)
        self._release_params()  # the module now holds the sharded parameters, whose .grad the clip and optimizer see
        for fp in unused:  # reached by the probe only (its zero-weighted backward): no gradient, as in TorchTrainer (H4 V16)
            fp.sharded_param.grad = None
        self.counters["probe_micro_batches"] += 1

    def _zero_fill_unsharded(self) -> list[Any]:
        """M12 audit H4 V2: FSDP2 reduce-scatters only the unsharded parameters that hold a gradient, so a parameter one
        rank's forward never reached (an idle expert) made the per-rank buffers differ and every parameter came out wrong
        (max |Δθ| = 2·lr). Agree on the set first: a parameter with a gradient on any rank gets a zero gradient where it
        has none; one with no gradient anywhere stays without (``TorchTrainer`` leaves it ``None`` too): returned, so the
        probe's zero gradient on it is dropped after the reduce-scatter."""
        fps = [fp for m in self._fsdp for fp in _fsdp_params(m)]
        if not fps:
            return []
        have = [fp.unsharded_accumulated_grad is not None or fp.unsharded_param.grad is not None for fp in fps]
        t = torch.tensor(have, dtype=torch.int32, device=self._coll_dev)
        self.dist.all_reduce(t, op=self.dist.ReduceOp.MAX, group=self.group)
        for fp, mine, anyone in zip(fps, have, t.tolist(), strict=True):
            if anyone and not mine:
                fp.unsharded_param.grad = torch.zeros_like(fp.unsharded_param)
                self.counters["zero_filled_grads"] += 1
        return [fp for fp, anyone in zip(fps, t.tolist(), strict=True) if not anyone]

    # ---- lockstep micro batches (layer mode) ----------------------------------------------------------------------
    def _max_count(self, n: int) -> int:
        t = torch.tensor([n], dtype=torch.int64, device=self._coll_dev)
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
        chunks = self._equalize([mb for p in self._local_parts(parts) for mb in self._split(p)])
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

    # ---- offload between steps (H4 V7 / G5 D3) ---------------------------------------------------------------------
    def _empty_cache(self) -> None:
        import gc

        gc.collect()
        if self.device.type == "cuda":
            if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
                torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()

    def offload(self, mode: str = "cpu", *, path: str | None = None) -> None:
        """``cpu``: this rank's parameter shards and AdamW moment shards to host memory, gradients dropped, the CUDA
        cache emptied (verl's FSDP2 offload: ``model.cpu()`` moves the DTensor shards, FSDP re-pads them). ``onload()``
        moves them back. A local operation (no collective); ``nvme`` is refused (host memory holds 16 B/param ÷ shards)."""
        if mode == "none" or getattr(self, "_offloaded", None):
            return
        if mode != "cpu":
            raise ValueError("FSDPTrainer offload: none | cpu (nvme is not supported for sharded state)")
        self._release_params()
        if self._host_resident:  # CPUOffloadPolicy: nothing to move; free what the step left in the caching allocator
            self._empty_cache()
            return
        for p in self.model.parameters():
            p.grad = None
        self._move("cpu")
        self._offloaded = mode
        self._empty_cache()

    def onload(self) -> None:
        if not getattr(self, "_offloaded", None):
            return
        self._move(self.device)
        self._offloaded = None

    def _move(self, dev: Any) -> None:
        params = list(self.model.parameters())
        state = [self.opt.state.get(p) for p in params]
        self.model.to(dev)  # FSDP2 ``_apply``: the shards move, the parameters stay the same objects (swap_tensors)
        now = list(self.model.parameters())
        if any(a is not b for a, b in zip(params, now, strict=True)):  # a torch that replaced them: re-key the optimizer state
            for old, new, st in zip(params, now, state, strict=True):
                self.opt.state.pop(old, None)
                if st is not None:
                    self.opt.state[new] = st
            for g in self.opt.param_groups:
                g["params"] = [now[params.index(p)] if p in params else p for p in g["params"]]
        for st in self.opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v) and k != "step":
                    st[k] = v.to(dev, non_blocking=False)


def _mp_policy(spec: Any) -> Any:
    """``"bf16"`` / ``"fp16"``: bf16 / fp16 gathered parameters, fp32 reduce-scatter (verl's FSDP2 default)."""
    if spec is None or spec is False:
        return None
    from torch.distributed.fsdp import MixedPrecisionPolicy

    if isinstance(spec, MixedPrecisionPolicy):
        return spec
    dt = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16}.get(str(spec))
    if dt is None:
        raise ValueError(f"mixed_precision: bf16 | fp16 | MixedPrecisionPolicy (got {spec!r})")
    return MixedPrecisionPolicy(param_dtype=dt, reduce_dtype=torch.float32)
