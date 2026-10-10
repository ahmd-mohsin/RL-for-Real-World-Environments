"""Distributed weight sync: the scale-out weight plane (research m11-m12/04 §A, checkpoint-engine PATTERN; §B.4 OpenRLHF;
m11-m12/06 §4.1).

``InprocPublisher`` / ``CheckpointPublisher`` (``weights.py``) cover one process and the universal disk path. Here the
trainer's weights reach engine processes over a ``torch.distributed`` group (NCCL on GPUs, gloo on CPU), following
Moonshot's checkpoint-engine protocol:

1. **Bucket plan** (``plan_buckets``): every parameter gets ``ParamMeta(name, dtype, shape, offset, nbytes, bucket,
   owner)``; tensors are sorted by (owner, name), offsets 256-byte aligned, and a bucket holds one owner's tensors
   only (the owner is the broadcast source). A tensor larger than the bucket is refused. Owners are the ranks that
   hold weights: one trainer rank, or several trainer ranks each owning a shard (colocated: an owner is also a
   receiver). The plan is gathered once (``all_gather_object`` of each owner's shapes) and is **portable**
   (``to_json`` / ``from_json``): a late engine learns every shard's location without the trainer (elastic join).
2. **Bucket size** (``choose_bucket_bytes``): ``all_reduce(MIN)`` of free memory × fraction across ranks; a third of
   it when the largest tensor fits there (room for a staging buffer), else half; capped by ``max_bucket``, never below
   the largest tensor.
3. **Double-buffered pipeline** (``BroadcastWeightSync.update``): a 2 × bucket byte buffer. Bucket *g* is staged by its
   owner into half *g mod 2* and broadcast; while bucket *g+1* broadcasts into the other half, receivers reload bucket
   *g* on a worker thread. Before half *g mod 2* is overwritten, the reload of bucket *g−2* has finished (at most one
   reload in flight).
4. **All-reduced error codes**: after every bucket the previous reload's status is ``all_reduce(MAX)``-ed; any rank
   failing aborts every rank with ``WeightSyncError``.

Crucible adds what checkpoint-engine leaves to the caller (research §A.6):

- **policy versions**: a header broadcast ``[version, plan crc]`` opens every update; receivers return the version
  with the weights, and a plan drift (shapes changed) is refused on every rank;
- **atomic commit**: receivers reload into a staging dict and ``commit(version, state)`` only after the last bucket
  succeeded everywhere, so a failed update leaves the old weights *and* the old version serving; an engine never
  serves a half-updated model (engines pin weights per call; ``TorchEngine.load_weights`` swaps the model object);
- **streaming while serving**: receiving does not stop generation (disaggregated topology); only the commit swaps,
  at a call boundary. The colocated caller still pauses / aborts and resets the prefix cache around a publish
  (``runtime.py``; vLLM ``/reset_prefix_cache``).
- **subgroup updates** (``ranks=[...]``): only the listed ranks take part (a restarted engine catches up while others
  keep serving); non-members return at once.

On GPUs the buffer lives on the device and the broadcast is NCCL; serving the received bytes to a separate vLLM
process zero-copy needs CUDA IPC (checkpoint-engine's ``VllmColocateWorkerExtension``), a later step: here receivers
commit through their engine's ``load_weights``.
"""

from __future__ import annotations

import json
import threading
import zlib
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import torch

ALIGN = 256
GiB = 1 << 30
DTYPES = {str(d).removeprefix("torch."): d for d in (torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int64, torch.int32,
                                                     torch.int16, torch.int8, torch.uint8, torch.bool)}


class WeightSyncError(RuntimeError):
    """A rank failed during an update; every rank raises it and no rank commits."""


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


@dataclass(frozen=True)
class ParamMeta:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int  # within its bucket
    nbytes: int
    bucket: int
    owner: int


@dataclass
class BucketPlan:
    bucket_bytes: int
    metas: list[ParamMeta]

    def buckets(self) -> list[list[ParamMeta]]:
        out: list[list[ParamMeta]] = []
        for m in self.metas:
            while len(out) <= m.bucket:
                out.append([])
            out[m.bucket].append(m)
        return out

    def bucket_owner(self, g: int) -> int:
        return self.buckets()[g][0].owner

    def used(self, g: int) -> int:
        b = self.buckets()[g]
        return _align(b[-1].offset + b[-1].nbytes)

    def crc(self) -> int:
        return zlib.crc32(self.to_json().encode()) & 0x7FFFFFFF

    def to_json(self) -> str:
        return json.dumps({"bucket_bytes": self.bucket_bytes, "metas": [asdict(m) for m in self.metas]}, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> BucketPlan:
        d = json.loads(text)
        return cls(d["bucket_bytes"], [ParamMeta(**{**m, "shape": tuple(m["shape"])}) for m in d["metas"]])


def _nbytes(dtype: str, shape: Iterable[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n * torch.empty((), dtype=DTYPES[dtype]).element_size()


def plan_buckets(shapes: dict[str, tuple[str, tuple[int, ...]]], bucket_bytes: int, owners: dict[str, int] | None = None) -> BucketPlan:
    """``shapes``: name → (dtype, shape); ``owners``: name → owner rank (default 0). Deterministic for equal inputs."""
    owners = owners or {}
    metas: list[ParamMeta] = []
    bucket, offset, cur_owner = -1, 0, None
    for name in sorted(shapes, key=lambda n: (owners.get(n, 0), n)):
        dtype, shape = shapes[name]
        nb = _nbytes(dtype, shape)
        if nb > bucket_bytes:
            raise ValueError(f"{name}: {nb} bytes exceeds the bucket ({bucket_bytes}); raise bucket_bytes to at least the largest tensor")
        owner = owners.get(name, 0)
        if owner != cur_owner or _align(offset) + nb > bucket_bytes:
            bucket, offset, cur_owner = bucket + 1, 0, owner
        off = _align(offset)
        metas.append(ParamMeta(name, dtype, tuple(int(s) for s in shape), off, nb, bucket, owner))
        offset = off + nb
    return BucketPlan(bucket_bytes, metas)


def choose_bucket_bytes(free_bytes_min: int, max_tensor: int, *, fraction: float = 0.9, max_bucket: int = 8 * GiB) -> int:
    """checkpoint-engine ``_detect_bucket_size``: usable = free × fraction; a third of it when the largest tensor fits
    (room for an H2D staging buffer beside the 2-bucket buffer), else half; at most ``max_bucket``, at least the largest
    tensor (else the update cannot run)."""
    usable = int(free_bytes_min * fraction)
    cap = usable // 3 if max_tensor <= usable // 3 else usable // 2
    b = min(max(max_bucket, max_tensor), cap)
    if b < max_tensor:
        raise ValueError(f"not enough free memory for a bucket: largest tensor {max_tensor} B, usable {usable} B")
    return _align(b)


class StagedState:
    """Receiver-side staging: bucket reloads copy into a fresh dict; ``take()`` hands the complete state to commit."""

    def __init__(self) -> None:
        self._d: dict[str, torch.Tensor] = {}

    def load(self, named: list[tuple[str, torch.Tensor]]) -> None:
        for n, t in named:
            self._d[n] = t.clone()

    def take(self) -> dict[str, torch.Tensor]:
        d, self._d = self._d, {}
        return d


class BroadcastWeightSync:
    """One participant of the weight group (``torch.distributed`` must be initialized). Every rank calls the same
    methods in the same order (collectives). Owners pass their tensors; receivers pass ``receive=True``."""

    def __init__(self, *, group: Any = None, device: str | torch.device = "cpu", bucket_bytes: int | None = None, free_bytes: int | None = None,
                 max_bucket: int = 8 * GiB, coordinator: int = 0):
        import torch.distributed as dist

        self.dist, self.group, self.device, self.coordinator = dist, group, torch.device(device), coordinator
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self._bucket_bytes, self._free, self._max_bucket = bucket_bytes, free_bytes, max_bucket
        self.plan_: BucketPlan | None = None
        self.version = 0
        self.stats: list[dict] = []

    # ---- plan --------------------------------------------------------------------------------------------------
    def plan(self, owned: dict[str, torch.Tensor] | None = None) -> BucketPlan:
        """Collective: gather every owner's (name, dtype, shape); choose the bucket size; build the global plan."""
        mine = {n: (str(t.dtype).removeprefix("torch."), tuple(t.shape)) for n, t in (owned or {}).items()}
        allv: list[Any] = [None] * self.world
        self.dist.all_gather_object(allv, mine, group=self.group)
        shapes: dict[str, tuple[str, tuple[int, ...]]] = {}
        owners: dict[str, int] = {}
        for r, d in enumerate(allv):
            for n, v in (d or {}).items():
                if n in owners:
                    raise ValueError(f"{n} owned by ranks {owners[n]} and {r}")
                shapes[n], owners[n] = (v[0], tuple(v[1])), r
        max_tensor = max((_nbytes(*v) for v in shapes.values()), default=1)
        b = self._bucket_bytes
        if b is None:
            free = self._free if self._free is not None else self._free_memory()
            t = torch.tensor([free], dtype=torch.int64)
            self.dist.all_reduce(t, op=self.dist.ReduceOp.MIN, group=self.group)
            b = choose_bucket_bytes(int(t.item()), max_tensor, max_bucket=self._max_bucket)
        self.plan_ = plan_buckets(shapes, b, owners)
        return self.plan_

    def load_plan(self, text: str) -> BucketPlan:
        """Elastic join: adopt a plan exported by a running member instead of gathering it."""
        self.plan_ = BucketPlan.from_json(text)
        return self.plan_

    def _free_memory(self) -> int:
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            return int(free)
        return 4 * GiB

    # ---- update ------------------------------------------------------------------------------------------------
    def update(self, version: int | None = None, owned: dict[str, torch.Tensor] | None = None, *, receive: bool = False,
               commit: Callable[[int, dict[str, torch.Tensor]], None] | None = None, ranks: list[int] | None = None,
               load: Callable[[list[tuple[str, torch.Tensor]]], None] | None = None) -> int | None:
        """Collective over the group (or the subgroup ``ranks``; every rank must call, non-members return None at once).
        The coordinator passes ``version``; receivers get it from the header. Returns the version committed (receivers),
        the version sent (pure owners), or None (non-members). ``load`` overrides staging (tests / custom engines)."""
        import time

        dist = self.dist
        group = self.group
        if ranks is not None:
            group = dist.new_group(sorted(set(ranks)))  # collective over the world, as torch requires
            if self.rank not in ranks:
                return None
        plan = self.plan_
        if plan is None:
            raise RuntimeError("call plan() (or load_plan()) before update()")
        t0 = time.monotonic()
        hdr = torch.tensor([version if version is not None else -1, plan.crc()], dtype=torch.int64)
        dist.broadcast(hdr, src=self.coordinator, group=group)
        ver = int(hdr[0])
        mismatch = torch.tensor([int(int(hdr[1]) != plan.crc())], dtype=torch.int64)
        dist.all_reduce(mismatch, op=dist.ReduceOp.MAX, group=group)
        if int(mismatch):
            raise WeightSyncError(f"bucket plans differ across ranks (shapes changed?): replan on every rank; rank {self.rank} crc {plan.crc()}")
        staged = StagedState()
        reload = load or staged.load
        B = plan.bucket_bytes
        buf = torch.empty(2 * B, dtype=torch.uint8, device=self.device)
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crucible-reload") if receive else None
        prev: Future | None = None
        err = {"msg": None}

        def run_reload(named: list[tuple[str, torch.Tensor]]) -> None:
            try:
                reload(named)
            except Exception as e:
                err["msg"] = f"rank {self.rank}: {type(e).__name__}: {e}"
                raise

        def check(fut: Future | None, g: int) -> None:
            bad = 0
            if fut is not None:
                try:
                    fut.result()
                except Exception:  # noqa: BLE001
                    bad = self.rank + 1
            code = torch.tensor([bad], dtype=torch.int64)  # MAX → the highest failing rank: every rank names the same culprit
            dist.all_reduce(code, op=dist.ReduceOp.MAX, group=group)
            if int(code):
                raise WeightSyncError(f"rank {int(code) - 1} failed to reload bucket {g} of version {ver}; no rank commits"
                                      + (f" ({err['msg']})" if err["msg"] else ""))

        try:
            buckets = plan.buckets()
            for g, metas in enumerate(buckets):
                half = buf[(g % 2) * B: (g % 2) * B + plan.used(g)]
                owner = metas[0].owner
                if owner == self.rank:
                    assert owned is not None, f"rank {self.rank} owns bucket {g} but passed no tensors"
                    for m in metas:
                        src = owned[m.name].detach().contiguous()
                        half[m.offset: m.offset + m.nbytes].copy_(src.reshape(-1).view(torch.uint8).to(self.device), non_blocking=True)
                    if self.device.type == "cuda":
                        torch.cuda.current_stream(self.device).synchronize()
                dist.broadcast(half, src=owner, group=group)
                check(prev, g - 1)  # the previous bucket's reload: done on every rank, or everyone aborts
                prev = None
                if receive:
                    named = [(m.name, half[m.offset: m.offset + m.nbytes].view(DTYPES[m.dtype]).view(m.shape)) for m in metas]
                    prev = pool.submit(run_reload, named)
            check(prev, len(buckets) - 1)
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        if receive:
            state = staged.take() if load is None else {}
            if commit is not None:
                commit(ver, state)
            self.version = ver
        dist.barrier(group=group)
        self.stats.append({"version": ver, "buckets": len(plan.buckets()), "bytes": sum(m.nbytes for m in plan.metas), "seconds": time.monotonic() - t0})
        return ver


# ---- the runtime's publisher contract over the group ----------------------------------------------------------
class DistributedPublisher:
    """``publish()`` on the trainer (coordinator, owner) rank: version + 1, the local engines (colocated) load the state
    directly, and the bucketed broadcast reaches every remote engine rank (``EngineReceiver``). The collective runs in
    a worker thread so the event loop (rollouts, environments) keeps running during the transfer."""

    def __init__(self, trainer: Any, sync: BroadcastWeightSync, *, engines: Any = None, post_update_transform: Callable[[dict], dict] | None = None,
                 reset_optimizer_on_publish: bool = False):
        self.trainer, self.sync, self.engines = trainer, sync, engines
        self.transform, self.reset = post_update_transform, reset_optimizer_on_publish
        self.version = 0
        self.latencies: list[float] = []
        self._lock = threading.Lock()

    async def publish(self) -> int:
        import asyncio
        import time

        t0 = time.monotonic()
        state = self.trainer.state_dict()
        if self.transform is not None:
            state = self.transform(state)
        self.version += 1
        if self.sync.plan_ is None:
            await asyncio.to_thread(self.sync.plan, state)
        await asyncio.to_thread(self._send, self.version, state)
        if self.engines is not None:
            await self.engines.load_weights(state, self.version)
        if self.reset:
            self.trainer.reset_optimizer()
        self.latencies.append(time.monotonic() - t0)
        return self.version

    def _send(self, version: int, state: dict) -> None:
        with self._lock:
            self.sync.update(version, state)


class EngineReceiver:
    """On an engine rank: receive each publish and commit it to the engine atomically (``load_weights(state, version)``
    at a call boundary). ``receive_once`` per publish; ``plan()`` must be called once, matching the trainer's."""

    def __init__(self, sync: BroadcastWeightSync, engine: Any):
        self.sync, self.engine = sync, engine

    async def receive_once(self) -> int:
        import asyncio

        box: dict[str, Any] = {}
        ver = await asyncio.to_thread(self.sync.update, None, None, receive=True, commit=lambda v, s: box.update(v=v, s=s))
        await self.engine.load_weights(box["s"], box["v"])
        return int(ver)
