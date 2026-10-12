# M12 infra: scale sweeps (batch C)

Reproduce: `uv run python scripts/infra_scale_sweep.py --steps 8 --out results.json` (about 10 minutes on a laptop CPU).
Raw results: `01-scale-sweeps.json`.

The simulator (`train/sim.py`) drives the **real** `Scheduler` and `DataPool` with a timing model, so these numbers
measure the scheduling rules as they run in the runtime. The engine model is aggregate: N engines × 16 sequence
slots, 50 tokens/s per sequence. Rollout shapes: 2–6 turns (12 for the long tail), 80–300 completion tokens per turn,
and 0.5–3 s of environment time per turn. 20% of rollouts are 8× longer. Groups have n = 16 members, are the training
unit, and need all members terminal. Training takes 30 s fixed plus 90 s over the batch's tokens, so 120 s per step at
every scale. Publish takes 5 s.

## 1. Control plane at DORA scale (τ = 2)

Slots = envs / 2 (a live rollout spends part of its life in environment steps). The admission cap S·B·n with S = 2 is
about the environment pool.

| engines | slots | envs | B | topology | s / step | engine util % | trainer idle s / step | max staleness | re-prefill Mtok | max in flight | control-plane wall s / step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 2048 | 4096 | 85 | colocated_sync | 581.2 | 6.9 | 489.8 | 0 | 0.0 | 1360 | 0.08 |
| 128 | 2048 | 4096 | 85 | colocated_partial | 327.9 | 13.7 | 237.0 | 2 | 12.1 | 2720 | 0.11 |
| 128 | 2048 | 4096 | 85 | disaggregated_stream | 271.8 | 17.3 | 180.4 | 2 | 0.0 | 2720 | 0.06 |
| 128 | 2048 | 4096 | 85 | elastic | 260.0 | 17.7 | 214.3 | 2 | 0.0 | 2720 | 0.06 |
| 512 | 8192 | 16384 | 341 | colocated_sync | 583.5 | 7.0 | 491.0 | 0 | 0.0 | 5456 | 0.59 |
| 512 | 8192 | 16384 | 341 | colocated_partial | 332.4 | 13.7 | 240.6 | 2 | 48.0 | 10912 | 0.84 |
| 512 | 8192 | 16384 | 341 | disaggregated_stream | 268.9 | 17.6 | 176.5 | 2 | 0.0 | 10912 | 0.39 |
| 512 | 8192 | 16384 | 341 | elastic | 257.6 | 17.9 | 211.3 | 2 | 0.0 | 10912 | 0.43 |
| 1024 | 16384 | 32768 | 682 | colocated_sync | 585.9 | 6.9 | 493.6 | 0 | 0.0 | 10912 | 1.71 |
| 1024 | 16384 | 32768 | 682 | colocated_partial | 331.6 | 13.7 | 240.1 | 2 | 94.2 | 21824 | 2.50 |
| 1024 | 16384 | 32768 | 682 | disaggregated_stream | 270.8 | 17.4 | 178.6 | 2 | 0.0 | 21824 | 1.04 |
| 1024 | 16384 | 32768 | 682 | elastic | 259.4 | 17.7 | 213.3 | 2 | 0.0 | 21824 | 1.09 |
| 2048 | 32768 | 32768 | 682 | colocated_sync | 585.9 | 3.5 | 493.6 | 0 | 0.0 | 10912 | 1.56 |
| 2048 | 32768 | 32768 | 682 | colocated_partial | 331.5 | 6.8 | 240.0 | 2 | 94.2 | 21824 | 2.45 |
| 2048 | 32768 | 32768 | 682 | disaggregated_stream | 270.8 | 8.7 | 178.6 | 2 | 0.0 | 21824 | 1.03 |
| 2048 | 32768 | 32768 | 682 | elastic | 259.4 | 8.9 | 213.3 | 2 | 0.0 | 21824 | 1.08 |

- The scheduling outcome is scale-invariant: the same per-step behaviour from 128 to 2,048 engines. The scheduler
  makes the same decisions at 22k live rollouts as at 2.7k.
- **Control-plane cost:** at most 2.5 wall seconds per simulated step at 22k in flight, about 1% of a 270 s step.
  Profiling showed the `wait` rule scanning every live group's members per decision (36M generator steps in 4 steps
  at 16k in flight), and `live_count` scanning every tag. The pool indexes in this batch cut the scheduler's cost
  7.7× at 2k in flight and then a further 4.2× at 16k in flight. The runtime's launch loop had the same tag scan and
  is indexed too. Outcomes are identical on the profiled runs (steps, simulated time, staleness), and the full test
  suite passes unchanged.
- Doubling engines beyond the environment pool (2,048 engines, 32k envs) halves utilization and buys nothing: the
  in-flight cap, not the engines, sets throughput.
- `colocated_partial` re-prefills 94M tokens per 8 steps at 1,024 engines: every barrier aborts every in-flight call.
  `resume_keeps_kv` (DeepSeek-V4.1) removes this term where the engine supports it (batch A).

## 2. The staleness window at 1,024 engines / 32k environments

τ = admission window S. Training takes 120 s per step (`train_scale` 1) or 480 s (4).

| τ | train scale | topology | s / step | engine util % | trainer idle s / step | max staleness | max in flight | re-prefill Mtok |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | colocated_partial | 585.9 | 6.9 | 493.6 | 0 | 10912 | 0.0 |
| 1 | 1 | disaggregated_stream | 581.5 | 7.0 | 489.3 | 1 | 10912 | 0.0 |
| 1 | 1 | elastic | 535.4 | 7.6 | 489.3 | 1 | 10912 | 0.0 |
| 2 | 1 | colocated_partial | 331.6 | 13.7 | 240.1 | 2 | 21824 | 94.2 |
| 2 | 1 | disaggregated_stream | 270.8 | 17.4 | 178.6 | 2 | 21824 | 0.0 |
| 2 | 1 | elastic | 259.4 | 17.7 | 213.3 | 2 | 21824 | 0.0 |
| 4 | 1 | colocated_partial | 222.3 | 24.4 | 130.5 | 4 | 43648 | 279.6 |
| 4 | 1 | disaggregated_stream | 167.0 | 36.4 | 69.5 | 4 | 43648 | 0.0 |
| 4 | 1 | elastic | 155.3 | 39.1 | 106.8 | 4 | 43648 | 0.0 |
| 8 | 1 | colocated_partial | 173.5 | 40.7 | 85.3 | 7 | 87296 | 687.2 |
| 8 | 1 | disaggregated_stream | 136.0 | 56.3 | 43.7 | 7 | 87296 | 0.0 |
| 8 | 1 | elastic | 89.8 | 77.2 | 43.7 | 7 | 87296 | 0.0 |
| 4 | 4 | colocated_partial | 497.8 | 10.9 | 130.5 | 4 | 43648 | 279.6 |
| 4 | 4 | disaggregated_stream | 475.4 | 14.9 | 44.8 | 4 | 43648 | 0.0 |
| 4 | 4 | elastic | 253.8 | 26.7 | 44.8 | 4 | 43648 | 0.0 |

- **For long-tail agentic rollouts the staleness bound is the throughput knob.** A group trains only when its slowest
  member ends, and with n = 16 and a 20% tail, 97% of groups contain a tail rollout. Going from τ = 1 to τ = 8 cuts
  the step from 582 s to 136 s (disaggregated) and lifts engine utilization from 7% to 56%. No trained rollout exceeds
  τ in any run (the B.21 rules hold at this scale).
- **Elastic pays off when training is the bottleneck.** At τ = 2 the run is rollout-bound, and elastic gains 4%. At
  τ = 8, or with training 4× heavier at τ = 4, the run is train-bound: elastic's group joining the trainer
  (`elastic_train_speedup` 2.0 here) gives 90 s vs 136 s and 254 s vs 475 s. This is DORA's design point.
- Colocated partial rollout pays the abort and re-prefill tax, which grows with τ (687M tokens at τ = 8), and trails
  disaggregated at every τ when the engines and trainer are separate devices. Its case is a cluster too small to split
  (see the planner, 8 nodes).

## 3. Placement (planner, `train/placement.py`)

H100-class nodes (8 × 80 GB, 989 dense TFLOPs, 450 GB/s NVLink, 50 GB/s network per GPU, 25 GB/s PCIe), MFU 0.35.
Workloads: 8B and 70B take 4,096 sequences per step, 2k-token prompts and 8k responses. The 1T MoE (32B active) takes
8,192 sequences, 4k / 16k tokens, 8 turns and 5 s of environment time per turn. Decode runs at 40 tokens/s per sequence.

| model | nodes | best | step s | engines × GPUs | trainer GPUs | elastic GPUs | mesh (rep, shard) | sync / partial / disagg / elastic |
|---|---|---|---|---|---|---|---|---|
| 8B dense | 8 | colocated_partial | 393 | 64 × 1 | 64 | 0 | (16, 4) | 1324 / 393 / 397 / 393 |
| 8B dense | 32 | disaggregated_stream | 205 | 224 × 1 | 32 | 0 | (8, 4) | 641 / 231 / 205 / 205 |
| 8B dense | 1024 | disaggregated_stream | 205 | 8160 × 1 | 32 | 0 | (8, 4) | 619 / 209 / 205 / 205 |
| 70B dense | 8 | colocated_partial | 1893 | 16 × 4 | 64 | 0 | (2, 32) | 4487 / 1893 / 2185 / 1932 |
| 70B dense | 32 | elastic | 477 | 62 × 4 | 8 | 120 | (4, 32) | 1434 / 478 / 533 / 477 |
| 70B dense | 128 | disaggregated_stream | 205 | 192 × 4 | 256 | 0 | (8, 32) | 670 / 260 / 205 / 205 |
| 1T MoE | 8, 32 | infeasible | – | – | – | – | – | 16 TB of trainer state needs ≥ 286 GPUs at 70% of 80 GB |
| 1T MoE | 128 | colocated_partial | 960 | 16 × 64 | 1024 | 0 | (2, 512) | 2598 / 960 / 1678 / 1001 |
| 1T MoE | 512 | disaggregated_stream | 450 | 56 × 64 | 512 | 0 | (1, 512) | 1301 / 482 / 450 / 450 |
| 1T MoE | 1024 | disaggregated_stream | 450 | 120 × 64 | 512 | 0 | (1, 512) | 1290 / 470 / 450 / 450 |

- The regimes match the article's map. Small clusters colocate: there are too few GPUs to split, so partial rollout
  is the best use of a shared pool. Mid-size clusters for big dense models go elastic (DORA). Large clusters
  disaggregate and become latency-bound: 205 s is one 8k-token sequence at 40 tokens/s, and no placement goes below
  it. The next lever there is decode speed (speculative decoding, MTP), not GPUs.
- A 1T MoE engine spans 8 nodes (64 GPUs, expert / pipeline parallel across nodes). A single node cannot hold its
  weights with 40% kept for KV.
- Fully synchronous is 3× slower at every size: the synchronous round waits for its tail.

## What stays open

- **NCCL at multi-GPU scale.** FSDP2, HSDP, DCP and the SPMD driver are validated on gloo multi-process (CPU). The
  NCCL paths (`set_force_sum_reduction_for_comms` off, PREMUL_SUM, CUDA meshes) need a multi-GPU node; the lab 5090 is
  a single GPU.
- **Layer mode needs one forward per micro batch.** R3 routing replay, prefix sharing and full-vocabulary OPD run in
  step mode only.
- **Restore of a replicated `DistributedTorchTrainer` under `SPMDDriver`** loads weights on the driver only. The
  sharded `FSDPTrainer` path is complete; replicas would need a broadcast after restore.
- **The planner is first order and uncalibrated.** It has no measured MFU, decode-speed curve vs batch, or KV
  pressure. Calibrate it from lab runs before trusting absolute seconds; the relative regime map is the robust output.
- **The simulator's engine is aggregate slots.** Per-engine routing, KV-cache pressure and prefill / decode
  interference are not modeled.
