# Live validation on real backends (2026-10-11)

Every check that needs real hardware or a real model ran on one of three backends:
- Modal multi-GPU (NCCL, real models, calibration);
- the lab node (Docker, KVM, simulators, attacks);
- OpenRouter (LLM policies and judges).

Raw results: `02-live-validation.json` (Modal) and `live/*.json` (lab, OpenRouter). Each finding that needed a code
change is a DESIGN.md entry (B.23–B.33).

## 1. Multi-GPU NCCL (Modal, 4 × L4, driver 610.57, torch 2.14.1+cu130, NCCL 2.30.7)

`scripts/modal_infra_check.py` re-runs the gloo multi-process scenarios with one process per GPU over NCCL and applies
the same assertions.

| scenario | ranks | result | soak |
|---|---|---|---|
| weight-sync broadcast (many buckets, every dtype, two versions) | 3 | pass, bitwise | 10 rounds |
| sharded owners / all-rank abort on a failed reload / subgroup + elastic plan / publisher → engine / slow reload | 2–3 | pass | 1 each |
| DP replicas = full-batch update (`dis`, `ppo`) | 2 | pass, replicas bitwise | 10 rounds |
| FSDP2 step and layer modes, two updates, skewed batch | 2 | pass, 1.2e-6 – 3.9e-6 vs single process | 5 rounds |
| HSDP 2 × 2 | 4 | pass, ≤ 3.3e-6 | 1 |
| HF Llama (3 blocks), layer mode | 2 | pass | 1 |
| `Runtime` over 4 HSDP ranks through `SPMDDriver`, DCP checkpoint and exact resume | 4 | pass (reward 0.05 → 0.24) | 1 |

Found and fixed:
- B.23: NCCL control tensors on the CPU, so the NCCL weight-sync path had never run;
- B.24: stale GRU weights under FSDP2 on CUDA (`flatten_parameters`);
- B.25: `TorchEngine` built its inputs on the CPU;
- B.26: Modal needs `NCCL_CUMEM_ENABLE=0`.

## 2. Real models, disaggregated, placed by the planner (Modal)

`crucible_pack_toy.m12.gate_dist` via `scripts/modal_gate_dist.py`:
- `placement.plan` → `cluster.process_specs` → `LocalLauncher`;
- vLLM engines on their own GPUs;
- `FSDPTrainer` ranks over NCCL driven by `SPMDDriver`;
- the real `Runtime` (`disaggregated_stream`), with `CheckpointPublisher` loading every engine each step.

| run | GPUs | model | split | steps | held-out success before → after | train reward | staleness | peak mem / rank |
|---|---|---|---|---|---|---|---|---|
| word-file task | 4 × A10G | Qwen3-0.6B | 2 engines + 2 FSDP | 20 / 20 | **0.44 → 1.00** (n = 96 each) | 0.52 → 1.00 | ≤ 1 | 7.3 GB |
| word-file task | 4 × A10G | Qwen3-1.7B | 2 engines + 2 FSDP | 9 (wall clock) | 1.00 → 1.00 | 1.00 | ≤ 1 | 18.2 GB |

The 1.7B run first ran out of memory at an 8192-token micro batch, although the planner said it fit (B.32). It ran
at 4096 tokens. The runtime's default 600 s wall clock stopped it at step 9; the gate now sets its own wall clock.
The task is already solved by 1.7B, so this run proves sharded training and publishing at that size, not learning.

## 3. Planner calibration (Modal, `scripts/modal_calibrate.py`)

GPU 0 serves with vLLM and GPUs 1–3 train (FSDP2 layer mode, gradient checkpointing, fp32 master + bf16 autocast, fp32
LM head).

| | A10G × 4, Qwen3-1.7B | H100 × 4, Qwen3-8B |
|---|---|---|
| decode per sequence at batch 1 / 8 / 32 / 128 (tok/s) | 111 / 106 / 86 / 53 | 141 / 142 / 134 / 96 |
| decode aggregate at batch 128 (tok/s) | 4,452 | 7,579 |
| training on 3 GPUs (tok/s) | 1,238 | 10,055 |
| MFU (vs dense bf16 peak) | 3.4 % | 16.7 % |
| peak memory per rank | 14.4 GB | 65.0 GB |
| weight-sync broadcast (Crucible's bucketed NCCL path) | 1.14 GB/s (PCIe) | 7.62 GB/s |
| bf16 checkpoint write | 4.1 GB in 1.3 s | 16.4 GB in 6.2 s |

The planner's defaults were optimistic on three counts:
- MFU 0.35;
- publish at raw NVLink speed;
- a state-only memory model.

It now models activations and the LM-head chunk with a fitted 1.25 overhead (within 10% of both peaks), uses the measured
decode-versus-batch curves, and ships both points as `placement.CALIBRATED`.

## 4. Lab node: KVM virtual machines (`common/vm`)

`tests/live/vm_check.py`, driven from this machine over `DOCKER_HOST=ssh://`:

| measure | value |
|---|---|
| cold boot to SSH, then `savevm ready` | 8.8 s + 0.6 s |
| a warm VM from the golden RAM snapshot (container + `-loadvm`) | 2.4–2.8 s (SSH 0.5 s after load) |
| reset (`loadvm ready`) | median 1.18 s, max 1.36 s |
| mid-episode World snapshot / restore (`savevm` / `loadvm`) | 0.6–0.7 s / 1.0–1.2 s; a background process and the files come back exactly |
| stress: 8 VMs × 64 concurrent acquire / exec / release | all clean, 0 files leaked between users |

Found and fixed: B.28, a pool reset task that died silently.

## 5. Lab node: Docker sandboxes (`tests/live/docker_stress.py`)

| phase | result |
|---|---|
| 48 concurrent cold hardened sandboxes (`--init`, no network, pids limit) | p50 0.50 s, p95 0.57 s, 0 failures |
| warm pool, fresh mode, 256 cycles at concurrency 32 | 5.0 s, 256 distinct containers, none handed out twice |
| warm pool, reuse mode with hygiene checks, 256 cycles | 224 reused, **0 leaks** (planted file, process and user never inherited), 0 dirty |
| cleanup | 0 containers left |

Before the fixes:
- B.29: per-call SSH hung 5 of 24 cold starts;
- B.30: reuse mode never reused.

The red-team attack suite passes on the local daemon (27/27) and in a container on the lab's own daemon (22/22; the 2
Harbor tests need Harbor in that container). Against a remote daemon from this machine it cannot run, because bind-mount
sources are host paths; `docker_grade` now says so (B.31).

## 6. Lab node: simulators

- **MuJoCo** (`gymnasium` HalfCheetah-v5 and Hopper-v5 as `common/control` Worlds):
  - a World rollout equals the raw gymnasium rollout exactly;
  - snapshot → restore replays 200 steps and reproduces rewards exactly;
  - 8 parallel Worlds run 1,311 steps/s;
  - PPO through Worlds on Hopper: **11 → 1,201** mean return over 500k steps in 208 s, greedy evaluation 1,040–1,277 (lab, 4 CPUs;
    the uncapped run spread torch over 22 cores at 2,200% CPU on a shared machine and was slower, so the runner now caps
    `--cpus 4`).
- **Unity ML-Agents** (3DBall through OpenEnv's Unity server in its own Python 3.10 container, reached by
  ControlApp's WebSocket client): reset 0.7 s, 8-dim observation, continuous actions. The random policy returns 0.36; PPO
  through Worlds over the remote link reaches 0.45 over 1,341 episodes. The server steps in real time (12 steps/s), so
  20k steps take 32 minutes. Found and fixed (B.33): session reuse, stochastic resets, a configurable keepalive,
  and the scene set once at server start.
- **CARLA**: not run. The server image (about 16 GB) would land on a shared disk at 96% use (78 GB free, falling); it
  needs the user's go-ahead or another host.

## 7. OpenRouter

| check | model | result | cost |
|---|---|---|---|
| group operators with real judges (`bt_pairwise`, `tournament`, `grs` → `gar`) | Claude Haiku 5.5 | known-good ranked first by every operator; failed answers stay 0 under GRS; GAR gives the cleaner correct answer more advantage | $0.001 |
| noise robustness, backlog (3 tickets) and word-file Worlds, 10 seeds × intensity 0 / 0.5 / 1.0 | GPT-6 Luna | noise never masks (0 of 60); backlog 0.57 / 0.63 / 0.63, word file 1.0 / 1.0 / 0.9 | $0.08 per pass |

Found and fixed: B.27, where instruction typos changed the exact word a task asked for. Unprotected, the word-file task
fell to 0.8 because the agent faithfully wrote the typo. With value tokens protected, the one remaining miss is real
ambiguity ("exactly THE yarrow" after a dropped word).

## Modal cost (estimated from GPU-minutes; budget $30)

| runs | ≈ $ |
|---|---|
| NCCL suite, soaks, FSDP diagnosis and reruns (L4 × 2–4) | 3.75 |
| real-model gates on A10G × 4 (0.6B, 1.7B OOM, 1.7B) | 2.95 |
| calibration A10G × 4 (including a failed first run) | 1.35 |
| calibration H100 × 4 | 4.00 |
| **total before the 8B run** | **≈ 12** |
