"""Cluster placement planner (M12-infra C; art. §2.1 colocated vs disaggregated, §2.3 elastic, §4 "scaling to
thousands of GPUs"). Given a cluster, a model and a workload, ``plan`` chooses the topology, the engine tensor
parallelism, the rollout / trainer split, the FSDP / HSDP mesh and the environment worker count, and emits a concrete
GPU-by-GPU layout plus the ``RuntimeConfig`` overrides and a ``SimConfig`` that reproduces the plan in the simulator.

The cost model is first order and every term is printed (``Placement.explain``):

- engine size: the smallest power-of-two TP inside a node (a TP group never crosses a node), else whole nodes (expert
  / pipeline parallel across nodes: 2, 4, 8 … nodes per engine), whose weight shard leaves at least ``kv_frac`` of each
  GPU for KV cache; sequences per engine = KV bytes / (KV bytes per token · context), capped;
- rollout time per step: synchronous rounds run ⌈sequences / slots⌉ waves, each ``tail_mult`` × the mean sequence;
  partial rollout and streaming are fluid (decode work / slots: continuous batching refills slots, the tail carries
  over); plus turns × env seconds;
- train time per step: 6 · active params · tokens / (GPUs · peak FLOPs · MFU); trainer memory 16 B per parameter
  (bf16 weights and grads, fp32 master and AdamW moments) sharded by FSDP; the shard group is the smallest power of
  two that fits, replicas (HSDP) fill the rest;
- publish: colocated reshards on NVLink (bytes / NVLink bandwidth); disaggregated broadcasts on the network (bucketed
  and pipelined: bytes / per-GPU network bandwidth, once per publish);
- step time: ``colocated_sync`` roll(all, tail) + train(all) + publish + swap; ``colocated_partial`` roll(all, fluid)
  + train(all) + publish + swap, where swap moves the trainer's state shard and the engine's weight shard over PCIe,
  both ways; ``disaggregated_stream`` max(roll(g_r), train(g_t) + publish) over node-granular splits;
  ``elastic`` T = max((work + s_e·t_e) / (s_r + s_e), t + publish, t_e) in slot-seconds: rollout slots s_r decode
  all step, the elastic slots s_e except for t_e = train + swap (the colocated role swap, on the elastic GPUs).

The simulator (``train.sim``) is the reference for scheduling effects (staleness, waste, tail); the planner is the
reference for where GPUs go. ``Placement.sim_config()`` joins them.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

TOPOLOGIES = ("colocated_sync", "colocated_partial", "disaggregated_stream", "elastic")


@dataclass(frozen=True)
class ClusterSpec:
    nodes: int
    gpus_per_node: int = 8
    gpu_mem_gb: float = 80.0
    gpu_tflops: float = 989.0  # dense bf16 peak (H100 SXM)
    nvlink_gbps: float = 450.0  # GB/s per GPU inside a node
    net_gbps: float = 50.0  # GB/s per GPU across nodes (400 Gb/s NIC per GPU)
    pcie_gbps: float = 25.0  # GB/s per GPU host ↔ device (colocated offload / onload)
    cpu_slots_per_node: int = 192  # environment worker slots per node (cores × oversubscription)

    @property
    def gpus(self) -> int:
        return self.nodes * self.gpus_per_node


@dataclass(frozen=True)
class ModelSpec:
    params_b: float
    layers: int
    kv_heads: int
    head_dim: int
    active_b: float | None = None  # MoE: parameters active per token (train FLOPs and decode cost); None = dense
    serve_bytes: int = 2  # bf16 serving weights and KV
    hidden: int | None = None  # model width: with ``vocab``, the trainer's activation memory is modeled (else state only)
    vocab: int | None = None

    @property
    def active(self) -> float:
        return self.active_b if self.active_b is not None else self.params_b

    @property
    def weight_bytes(self) -> float:
        return self.params_b * 1e9 * self.serve_bytes

    @property
    def kv_bytes_per_token(self) -> float:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.serve_bytes


@dataclass(frozen=True)
class Workload:
    seqs_per_step: int  # trained sequences per optimizer step (B·n)
    prompt_tokens: int = 2048
    response_tokens: int = 2048
    turns: int = 1
    env_s_per_turn: float = 0.0
    decode_tok_s: float = 40.0  # per sequence, at full engine batch (used when no ``decode_curve`` is given)
    # measured per-sequence decode speed against the engine's concurrent batch, ((batch, tok/s), ...): speed falls as the
    # batch grows (Modal A10G, Qwen3-1.7B: 111 at 1, 86 at 32, 53 at 128), so the engine's actual load sets it
    decode_curve: tuple[tuple[int, float], ...] | None = None
    max_seqs_per_engine: int = 256
    tail_mult: float = 3.0  # the slowest sequence of a synchronous round, relative to the mean
    mfu: float = 0.35
    kv_frac: float = 0.4  # share of engine GPU memory reserved for KV cache
    train_mem_frac: float = 0.7  # share of trainer GPU memory for weights, grads, optimizer state (and activations when modeled)
    micro_tokens: int = 8192  # tokens per trainer micro batch (TrainerConfig.token_budget): the activation term
    logit_chunk: int = 1024  # LM-head rows per chunk (HFPolicy.logit_chunk), held in fp32
    # allocator + FSDP buffers (gathered layers, reduce-scatter staging) on top of state + activations when the activation
    # term is modeled: measured 1.18 (A10G, 1.7B, 3 ranks) and 1.27 (H100, 8B, 3 ranks)
    mem_overhead: float = 1.25
    env_slots_per_seq: float = 1.0  # CPU worker slots one live rollout needs

    @property
    def context(self) -> int:
        return self.prompt_tokens + self.response_tokens

    def decode_at(self, batch: float) -> float:
        """Per-sequence decode tokens/s at a concurrent batch (log-batch interpolation of ``decode_curve``, clamped)."""
        if not self.decode_curve:
            return self.decode_tok_s
        pts = sorted(self.decode_curve)
        b = max(1.0, float(batch))
        if b <= pts[0][0]:
            return pts[0][1]
        for (b0, v0), (b1, v1) in itertools.pairwise(pts):
            if b <= b1:
                f = (math.log(b) - math.log(b0)) / (math.log(b1) - math.log(b0))
                return v0 + f * (v1 - v0)
        return pts[-1][1]

    @property
    def tokens(self) -> int:
        return self.seqs_per_step * self.context


@dataclass
class Placement:
    topology: str
    cluster: ClusterSpec
    engine_gpus: int  # GPUs per engine (TP inside a node; whole nodes when it spans several)
    engines: int
    seqs_per_engine: int
    rollout_gpus: int
    train_gpus: int
    elastic_gpus: int
    mesh: tuple[int, int]  # (replicate, shard): HSDP; replicate == 1 is plain FSDP
    env_workers: int
    times: dict[str, float]
    layout: list[list[str]] = field(default_factory=list)  # [node][gpu] → "engine:k/tp_rank" | "train:r" | "both:k/r" | "elastic:k/r"
    alternatives: dict[str, float] = field(default_factory=dict)  # topology → its best step time

    @property
    def step_s(self) -> float:
        return self.times["step"]

    def runtime_overrides(self) -> dict[str, Any]:
        return {"topology": self.topology, "elastic_engines": self.elastic_gpus // self.engine_gpus if self.topology == "elastic" else 0,
                "env_workers": self.env_workers}

    def trainer_kwargs(self) -> dict[str, Any]:
        """For ``FSDPTrainer``: the mesh shape (build with ``init_device_mesh(dev, mesh, mesh_dim_names=("replicate", "shard"))``)."""
        return {"mesh_shape": self.mesh, "dp": self.mesh[0] * self.mesh[1], "reshard": "layer" if self.mesh[1] > 1 else "step"}

    def sim_config(self, w: Workload, **kw: Any):
        """The simulator's view of this plan: aggregate slots, per-sequence decode, train time per kilo-token, publish."""
        from crucible.train.sim import SimConfig

        per_ktok = self.times["train"] / max(w.tokens / 1000.0, 1e-9)
        return SimConfig(topology=self.topology, slots=self.engines * self.seqs_per_engine, tok_per_s=w.decode_tok_s,
                         env_workers=self.env_workers, train_s_base=0.0, train_s_per_ktok=per_ktok, publish_s=self.times["publish"],
                         elastic_frac=self.elastic_gpus / max(self.rollout_gpus + self.elastic_gpus, 1),
                         elastic_train_speedup=(self.train_gpus + self.elastic_gpus) / self.train_gpus if self.topology == "elastic" else 1.0, **kw)

    def explain(self) -> str:
        t = self.times
        return (f"{self.topology}: {self.engines} engines × {self.engine_gpus} GPUs ({self.seqs_per_engine} seqs each) on {self.rollout_gpus} GPUs; "
                f"trainer {self.train_gpus} GPUs mesh {self.mesh}; elastic {self.elastic_gpus}; env workers {self.env_workers}; "
                f"rollout {t['rollout']:.1f}s, train {t['train']:.1f}s, publish {t['publish']:.2f}s → step {t['step']:.1f}s")


# Measured on real hardware (``scripts/modal_calibrate.py``; ``docs/research/m12-infra/02-live-validation.md``). ``mfu`` is
# this stack's (FSDP2 layer mode, gradient checkpointing, fp32 master + bf16 autocast, fp32 LM head) at small batches:
# the planner's default 0.35 is a well-tuned large-batch figure, an order of magnitude above what was measured here.
CALIBRATED: dict[str, dict] = {
    "a10g/qwen3-1.7b": {"cluster": {"gpu_mem_gb": 23.7, "gpu_tflops": 125.0, "nvlink_gbps": 1.14, "net_gbps": 1.14},
                         "workload": {"mfu": 0.034, "decode_curve": ((1, 110.8), (8, 106.1), (32, 86.0), (128, 53.2))},
                         "measured": {"train_tok_s_3gpu": 1237.7, "checkpoint_gb_s": 3.1, "broadcast_gb_s": 1.14, "peak_mem_gb": 14.4}},
    # the publish bandwidth is Crucible's bucketed weight sync (double-buffered broadcast + reload), not raw NVLink
    "h100/qwen3-8b": {"cluster": {"gpu_mem_gb": 80.0, "gpu_tflops": 989.0, "nvlink_gbps": 7.62},
                      "workload": {"mfu": 0.167, "decode_curve": ((1, 141.2), (8, 142.5), (32, 134.1), (128, 95.9))},
                      "measured": {"train_tok_s_3gpu": 10054.9, "checkpoint_gb_s": 2.6, "broadcast_gb_s": 7.62, "peak_mem_gb": 65.0}},
}


class PlacementError(ValueError):
    pass


def _pow2_upto(n: int) -> list[int]:
    out, k = [], 1
    while k <= n:
        out.append(k)
        k *= 2
    return out


def engine_shape(c: ClusterSpec, m: ModelSpec, w: Workload) -> tuple[int, int]:
    """(GPUs per engine, sequences per engine): the smallest node-local TP, else the fewest whole nodes, that leaves
    ``kv_frac`` of memory for KV."""
    mem = c.gpu_mem_gb * 1e9
    sizes = [t for t in _pow2_upto(c.gpus_per_node) if c.gpus_per_node % t == 0] + [k * c.gpus_per_node for k in _pow2_upto(c.nodes)[1:]]
    for g in sizes:
        if m.weight_bytes / g <= (1 - w.kv_frac) * mem * 0.95:
            kv = (mem * 0.95 - m.weight_bytes / g) * g  # the whole group's free memory holds KV
            seqs = min(w.max_seqs_per_engine, int(kv // (m.kv_bytes_per_token * w.context)))
            if seqs >= 1:
                return g, seqs
    raise PlacementError(f"{m.params_b}B does not fit for serving: weights {m.weight_bytes / 1e9:.0f} GB over at most "
                         f"{c.gpus}×{c.gpu_mem_gb} GB with {w.kv_frac:.0%} kept for KV")


def activation_bytes(m: ModelSpec, w: Workload) -> float:
    """Per-rank activation memory of one micro batch with gradient checkpointing (0 when width / vocab are unknown):
    the checkpointed layer inputs and one layer's recompute (``micro_tokens · hidden · (4 · layers + 24)`` bytes, bf16 with
    fp32 accumulators) plus the fp32 LM-head chunk with its softmax and gradient (``3 · logit_chunk · vocab · 4``). Fitted
    on Qwen3-1.7B under FSDP2 layer mode (Modal A10G: 17.5 GB allocated at 8192 tokens against 13.6 GB of state)."""
    if not (m.hidden and m.vocab):
        return 0.0
    return w.micro_tokens * m.hidden * (4 * m.layers + 24) + 3 * w.logit_chunk * m.vocab * 4


def train_mesh(c: ClusterSpec, m: ModelSpec, w: Workload, gpus: int) -> tuple[int, int]:
    """(replicate, shard): the smallest power-of-two shard group whose 16 B/param state (plus the activation term when the
    model's width and vocab are given) fits, HSDP replicas beyond it."""
    need = 16 * m.params_b * 1e9
    act = activation_bytes(m, w)
    over = w.mem_overhead if act else 1.0
    # with activations and overhead modeled, the budget is nearly the whole card (the measured 8B peak was 65 of 80 GB at
    # 3 ranks); a state-only estimate keeps the conservative fraction for what it does not count
    cap = (0.92 if act else w.train_mem_frac) * c.gpu_mem_gb * 1e9
    for shard in (d for d in range(1, gpus + 1) if gpus % d == 0):  # any divisor: 3 ranks shard 3 ways (FSDP needs no power of two)
        if (need / shard + act) * over <= cap:
            return gpus // shard, shard
    raise PlacementError(f"trainer state {need / 1e12:.1f} TB does not fit {gpus} GPUs at {w.train_mem_frac:.0%} of {c.gpu_mem_gb} GB")


def rollout_s(c: ClusterSpec, m: ModelSpec, w: Workload, gpus: int, *, tail: bool) -> float:
    """``tail`` (synchronous rounds): ⌈sequences / slots⌉ waves, each as long as its slowest member (``tail_mult`` × the
    mean). Otherwise fluid (continuous batching refills a slot the moment a sequence ends; partial rollout and streaming
    carry the tail into the next step): decode work / slots, never below one sequence's duration."""
    eg, seqs = engine_shape(c, m, w)
    slots = gpus // eg * seqs
    if slots < 1:
        return math.inf
    load = min(seqs, math.ceil(w.seqs_per_step / (gpus // eg)))  # sequences each engine actually decodes at once
    one = w.response_tokens / w.decode_at(load)
    roll = math.ceil(w.seqs_per_step / slots) * one * w.tail_mult if tail else max(one, w.seqs_per_step * one / slots)
    return roll + w.turns * w.env_s_per_turn


def train_s(c: ClusterSpec, m: ModelSpec, w: Workload, gpus: int) -> float:
    if gpus < 1:
        return math.inf
    return 6 * m.active * 1e9 * w.tokens / (gpus * c.gpu_tflops * 1e12 * w.mfu)


def publish_s(c: ClusterSpec, m: ModelSpec, *, colocated: bool) -> float:
    return m.weight_bytes / ((c.nvlink_gbps if colocated else c.net_gbps) * 1e9)


def _env_workers(c: ClusterSpec, w: Workload, engines: int, seqs: int) -> int:
    return int(min(c.nodes * c.cpu_slots_per_node, math.ceil(engines * seqs * w.env_slots_per_seq)))


def _layout(c: ClusterSpec, topology: str, eg: int, rollout_gpus: int, train_gpus: int, elastic_gpus: int) -> list[list[str]]:
    """Node-major GPU roles. Colocated: every GPU runs both. Otherwise trainer nodes first, then elastic, then rollout;
    engine groups are contiguous (counts are multiples of the engine size and blocks start node aligned), so a TP group
    stays inside one node and a multi-node engine owns whole nodes."""
    out = [["" for _ in range(c.gpus_per_node)] for _ in range(c.nodes)]
    flat = [(n, g) for n in range(c.nodes) for g in range(c.gpus_per_node)]
    if topology.startswith("colocated"):
        for i, (n, g) in enumerate(flat):
            out[n][g] = f"both:{i // eg}/{i}"
        return out
    i = 0
    for r in range(train_gpus):
        n, g = flat[i]
        out[n][g] = f"train:{r}"
        i += 1
    k = 0
    for role, count in (("elastic", elastic_gpus), ("engine", rollout_gpus)):
        for j in range(count):
            n, g = flat[i]
            out[n][g] = f"{role}:{k + j // eg}/{j % eg}" + (f"/{train_gpus + j}" if role == "elastic" else "")
            i += 1
        k += count // eg
    return out


def _candidate(c: ClusterSpec, m: ModelSpec, w: Workload, topology: str, rollout_gpus: int, train_gpus: int, elastic: int) -> Placement | None:
    eg, seqs = engine_shape(c, m, w)
    try:
        mesh = train_mesh(c, m, w, train_gpus + elastic)
    except PlacementError:
        return None
    if topology.startswith("colocated"):
        r = rollout_s(c, m, w, c.gpus, tail=topology == "colocated_sync")
        t = train_s(c, m, w, c.gpus)
        p = publish_s(c, m, colocated=True)
        # the role swap, each way, per GPU over PCIe: the trainer's state shard leaves while the engine's weights return
        swap = 2 * (16 * m.params_b * 1e9 / mesh[1] + m.weight_bytes / eg) / (c.pcie_gbps * 1e9)
        times = {"rollout": r, "train": t, "publish": p, "swap": swap, "step": r + t + p + swap}
        engines = c.gpus // eg
    else:
        p = publish_s(c, m, colocated=False)
        t = train_s(c, m, w, train_gpus + elastic)
        r = rollout_s(c, m, w, rollout_gpus, tail=False)
        times = {"rollout": r, "train": t, "publish": p, "step": max(r, t + p)}
        if topology == "elastic":  # slot-seconds: rollout slots all step long, elastic slots except for t_e with the trainer
            swap = 2 * (16 * m.params_b * 1e9 / mesh[1] + m.weight_bytes / eg) / (c.pcie_gbps * 1e9)  # as colocated, on the elastic GPUs
            t_e = t + swap
            _, seqs_e = engine_shape(c, m, w)
            s_r, s_e = rollout_gpus // eg * seqs_e, elastic // eg * seqs_e
            speed = w.decode_at(min(seqs_e, math.ceil(w.seqs_per_step / max(1, (rollout_gpus + elastic) // eg))))
            work = w.seqs_per_step * w.response_tokens / speed
            one = w.response_tokens / speed
            T = max((work + s_e * t_e) / (s_r + s_e), one) + w.turns * w.env_s_per_turn
            step = max(T, t + p, t_e)
            times.update({"rollout": T, "swap": swap, "elastic_roll_share": max(0.0, (step - t_e) / step), "step": step})
        engines = (rollout_gpus + elastic) // eg
    if not math.isfinite(times["step"]):
        return None
    return Placement(topology=topology, cluster=c, engine_gpus=eg, engines=engines, seqs_per_engine=seqs,
                     rollout_gpus=c.gpus if topology.startswith("colocated") else rollout_gpus,
                     train_gpus=c.gpus if topology.startswith("colocated") else train_gpus, elastic_gpus=elastic, mesh=mesh,
                     env_workers=_env_workers(c, w, engines, seqs), times=times)


def candidates(c: ClusterSpec, m: ModelSpec, w: Workload, topology: str) -> list[Placement]:
    """Every feasible placement of one topology (disaggregated / elastic: node-granular trainer splits, engine-granular
    on a single node; elastic GPUs in whole engines)."""
    if topology not in TOPOLOGIES:
        raise ValueError(f"topology: {' | '.join(TOPOLOGIES)} (got {topology!r})")
    eg, _ = engine_shape(c, m, w)
    if topology.startswith("colocated"):
        p = _candidate(c, m, w, topology, c.gpus, c.gpus, 0)
        return [p] if p else []
    out = []
    G = c.gpus
    gran = c.gpus_per_node if c.nodes > 1 else eg  # trainer splits: whole nodes; inside a single node, whole engines
    for k in range(1, G // gran):
        g_t = k * gran
        rest = G - g_t
        elastics = [0] if topology == "disaggregated_stream" else range(eg, rest - eg + 1, eg)
        for e in elastics:
            g_r = (rest - e) // eg * eg
            if g_r < eg or g_r != rest - e:  # every GPU gets a role: a split that strands GPUs is not a placement
                continue
            p = _candidate(c, m, w, topology, g_r, g_t, e)
            if p:
                out.append(p)
    return out


def plan(c: ClusterSpec, m: ModelSpec, w: Workload, topology: str | None = None) -> Placement:
    """The fastest feasible placement (of ``topology``, or over all four); ``alternatives`` lists each topology's best."""
    best: dict[str, Placement] = {}
    for topo in ([topology] if topology else TOPOLOGIES):
        cs = candidates(c, m, w, topo)
        if cs:
            best[topo] = min(cs, key=lambda p: (p.step_s, p.train_gpus))
    if not best:
        raise PlacementError(f"no feasible placement on {c.nodes}×{c.gpus_per_node} GPUs for {m.params_b}B")
    win = min(best.values(), key=lambda p: p.step_s)
    tr = 0 if win.topology.startswith("colocated") else win.train_gpus
    win.layout = _layout(c, win.topology, win.engine_gpus, win.rollout_gpus, tr, win.elastic_gpus)
    win.alternatives = {k: v.step_s for k, v in best.items()}
    return win
