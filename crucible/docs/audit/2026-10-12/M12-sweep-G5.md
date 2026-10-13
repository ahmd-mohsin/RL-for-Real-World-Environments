# M12 sweep, group G5: TRL, OpenRLHF, checkpoint-engine, and multi-GPU readiness

- **Crucible HEAD:** 3b522cb.
- **Sources** (`RL-Environments/06-frameworks-and-env-code/`):
  - `infra-huggingface-trl` @ 72cf37d, Apache-2.0;
  - `infra-openrlhf-openrlhf` @ dc2a7ad, Apache-2.0;
  - `infra-moonshotai-checkpoint-engine` @ 84631a2, MIT (`LICENCE`);
  - configs read from verl, slime, NeMo-RL and SkyRL (`examples/`, `scripts/`).
- **Probes** live in the session scratchpad, `m12/g5/`. Nothing under `crucible/src` or `crucible/tests` was edited.
- **Shorthand in this document:**
  - `T` = `trl/trainer/grpo_trainer.py`, `C` = `trl/trainer/grpo_config.py`, `VG` = `trl/generation/vllm_generation.py`;
  - `EM` = `openrlhf/trainer/ppo_utils/experience_maker.py`;
  - `CE` = `checkpoint_engine/ps.py`.
- **[ran]** marks a result measured in this sweep:
  - environment: torch 2.14.1 on CPU, gloo, 14 cores;
  - TRL and OpenRLHF code is copied verbatim into pure-torch modules (`trl_loss.py`, `openrlhf_algos.py`), with a file:line on every block.

> **The cluster fact that reshapes the plan.** REPACSS has **4 × H100 NVL 94 GB per node** (`scripts/repacss.sh:2`). It does not have 8 GPUs per node.
> - Eight H100s there means **two nodes**: every 8-GPU run is multi-node. Crucible has never run multi-node.
> - Its launchers cannot start a multi-node job under Slurm (D5).
> - NVL cards are NVLink-bridged **in pairs** and use PCIe Gen5 otherwise. TP=2 engines and FSDP shard groups must respect the bridge topology (`nvidia-smi topo -m`).

## 0. Headline defects

| # | Sev | Defect | Crucible file:line | Evidence |
|---|---|---|---|---|
| D1 | **P0** | **Any data-parallel rank with zero items hangs or aborts the step.** `step()` builds the metric all-reduce from this rank's own metric keys: `[loss, n_mb] + per-key sums`. A rank with no micro batches sends a length-2 tensor while the other ranks send 2+K. The gradient collectives match, so the hang lands on the metric reduce. Triggers: items < dp. That happens with budget barriers (`min_groups=1`), heavy masking, or small B·n at 8 ranks. The same code serves `DistributedTorchTrainer` and `FSDPTrainer`. | `trainer.py:655-656`, `:719-720` | **[ran]** `_edge_worker.py few_items`:<br>- 7 items on 8 gloo ranks, both `step` and `layer`: ranks 0-2 sit in `_reduce_scalars` (trainer.py:720) while ranks 3-7 are already in `state_dict` (fsdp.py:168). Deadlock until the timeout.<br>- 8 items: passes (diff 4.0e-6).<br>- 1 item on 2 ranks: gloo aborts with `EnforceNotMet op.nread == op.preamble.nbytes`.<br>- On NCCL a count mismatch is a hang until the watchdog fires, or silent garbage. |
| D2 | **P0** | **The planned layout for Qwen3-30B-A3B cannot be built.** `FSDPTrainer` shards a model that is already fully materialized on every rank's device. Before that, the parent `__init__` broadcasts every full parameter, and the gate loads fp32 to `cuda` on every rank. 30.5B × 4 B = 122 GB per GPU, above 80 or 94 GB. There is no meta-device init and no rank-0 load + `broadcast_from_rank0`. Host RAM also breaks: 4 ranks × 122 GB of fp32 `from_pretrained` per node. | `fsdp.py:71,88-91`; `trainer.py:702-704`; `packs/_toy/.../gate_dist.py:65` | **[ran]** `placement_repacss.py`: on 2×4 H100 NVL the planner returns `colocated_partial TP2×4 mesh (1, 8) layer` for 30B-A3B. Its 61 GB state shard "fits" 0.92 × 94 GB, but the ranks can never reach the sharded state. |
| D3 | **P0** | **Colocated vLLM + `FSDPTrainer` on one GPU cannot fit as planned.**<br>- `FSDPTrainer.offload("cpu"/"nvme")` raises `NotImplementedError`.<br>- With `train_state_on_rollout="none"`, the fp32 shards stay resident: 16 B/param ÷ shard. Nothing calls `torch.cuda.empty_cache()` after a step; the only call is in `TorchTrainer.offload`.<br>- The trainer process's caching allocator therefore keeps its training peak while vLLM `/wake_up` asks for KV memory.<br>- The planner's colocated cost model assumes the trainer state leaves the GPU, and it picks colocated for 8B on 1×8 H100 and 1×4 H100 NVL. | `fsdp.py:209-214`; `trainer.py:172-177`; `runtime.py:678-680,742-743`; `placement.py:300` | **[derived]** Qwen3-8B, 1×8, mesh (2,4) **[ran]**. Resident shards: 8.19e9 × 16 / 4 = 33 GB. Measured training peak at 3 ranks: 65 GB (`02-live-validation.md` §3). vLLM: 16 GB of weights + 0.4 × 80 GB of KV. 33 + 16 + 32 = 81 GB > 80 before any cached block. Colocated FSDP+vLLM has never run on a GPU; the 5090 run was a single-process `TorchTrainer`. |
| D4 | **P1** (latent P0) | **The TRL bridge does not run against TRL 72cf37d.**<br>(a) `trl_gen` calls `float(x)` on TRL's nested `(T, num_logprobs)` logprobs (VG:567-575; TRL itself takes `lp[0]`, T:1743).<br>(b) TRL's `RepeatSampler` already repeats each prompt `num_generations` times (T:1270-1277), and the bridge loops `num_generations` again, so rows = P×G². TRL requires one row per received prompt (T:1590-1596).<br>(c) **Latent P0:** masked episodes are dropped from `completion_ids` but kept as `None` in `crucible_reward`, so rewards shift onto the wrong completions.<br>(d) Per-call `asyncio.to_thread(vllm_generation.generate)` runs concurrent offline-`LLM` calls from threads. In server mode and colocate TP>1, `generate` is a collective (`gather_object` / `all_gather_object`, VG:594, :690), so per-rank call counts that differ deadlock.<br>The test is vacuous: it calls `rf(["p0"])` with a stub and never meets TRL's row contract. | `bridges/trl_bridge.py:24-26,36,46-50`; `tests/unit/test_train_bridges.py:64-98` | **[ran]** `trl_bridge_probe.py`:<br>- (a) `TypeError: float() argument must be ... not 'list'`;<br>- (b) 6 TRL rows in → 10 completion rows and 12 reward rows out (TRL needs 6);<br>- (c) with `num_generations=1` and pB masked: 2 completions against rewards `[1.0, None, 0.5]`, so pC's completion is paired with `None`. |
| D5 | P1 | **No multi-node launch under Slurm.**<br>- `LocalLauncher` spawns *every* spec, including other nodes', on the local host.<br>- `master_addr` defaults to `127.0.0.1`.<br>- `RayLauncher` needs a Ray cluster, and no sbatch/srun script exists.<br>- On REPACSS, 8 GPUs means 2 nodes, so this blocks every 8-GPU run. | `cluster.py:49,107-136,146-178` | Code. `scripts/repacss.sh` only opens interactive sessions. |
| D6 | P1 | **`LOCAL_RANK` holds the physical GPU index while `CUDA_VISIBLE_DEVICES` narrows the process to that one GPU.** Any torchrun-convention code (accelerate, TRL, DeepSpeed, vLLM `external_launcher`) calls `set_device(LOCAL_RANK)` and fails with "invalid device ordinal" on GPUs 1-7. Crucible's own gate uses `set_device(0)`. The unit workers use `cuda:{RANK}` (`_fsdp_worker.py:68`, `_weightsync_worker.py:46`), which is invalid on the second node. | `cluster.py:73-76` | **[ran]** `process_specs` env for trainer rank 1: `{'CUDA_VISIBLE_DEVICES': '1', 'LOCAL_RANK': '1', 'RANK': '1', 'MASTER_ADDR': '127.0.0.1'}`. |
| D7 | P1 | **vLLM has no NCCL weight path.** `DistributedPublisher` hands engines a state **dict** (`engines.load_weights(state, v)`). `VLLMServerEngine.load_weights` takes a **directory path**. `EngineReceiver` cannot run inside vLLM, so the NCCL weight sync validated on L4 never reaches vLLM. The only working vLLM publish path is `CheckpointPublisher` (disk), yet the planner's publish cost assumes a broadcast. | `weightsync.py:327-328`; `vllm_http.py:129-133`; `placement.py:256-257` | Code. Disk measured at 2.6 GB/s on H100: 16.4 GB of bf16 in 6.2 s. |
| D8 | P1 | **`CheckpointPublisher` never deletes `v{n}/`.** Qwen3-8B costs 16.4 GB per publish in bf16, or 33 GB without the gate's bf16 transform. That is 1.6-3.3 TB per 100 steps, and the run dies on quota or a full disk. | `weights.py:57-72` | Code. No `rmtree` or GC anywhere in `train/` or `gate_dist.py`. |
| D9 | P1 | **Micro batches budget unpadded tokens, but the forward right-pads to the longest sequence.** There is no varlen or packing path (verl `use_remove_padding`, NeMo-RL `sequence_packing`), so peak memory exceeds `token_budget`. This explains part of the 1.7B OOM at 8192 tokens (B.32). | `batching.py:50-64`; `hf.py:124-131` | **[ran]** `padding_probe.py`, 256 sequences on 8 ranks: a bimodal mix (85% × 600 tokens, 15% × 14k) gives a **3.42× padded/budget** micro batch and **233%** padding at budget 16384. Lognormal agentic lengths give 2.0×. |
| D10 | P1 | **The planner's FSDP shard groups cross node boundaries with no communication term.** `train_s` is FLOPs only. On 4-GPU nodes it picks ZeRO-3 meshes (1,7), (1,12), (1,13) that all-gather and reduce-scatter per layer over the inter-node link. That contradicts HSDP's "shards inside a node" (`fsdp.py:4-5`). `elastic` wins those plans, yet no vLLM elastic engine exists. The refusal message says "70%" while the cap is 92% (`placement.py:229` vs `:233`). | `placement.py:221-233,250-253,339-350` | **[ran]** `placement_repacss.py`: 2×4 → 8B `elastic mesh (1, 7)`; 4×4 → 4B `mesh (1, 13)`, 30B-A3B `disaggregated mesh (1, 12)`. |
| D11 | P1 | **Every aggregation except `token_mean_global` runs the whole local batch as one forward.** The micro-batch path is skipped (`single` → `mbs=[mine]`), and `DistributedTorchTrainer` refuses them, so at 8 GPUs only token-mean is usable. TRL `grpo` and `dr_grpo` parity (both match exactly, §1) is single-GPU only and OOMs at real batch sizes. | `trainer.py:606-607,625,700-701` | Code. |

**P2 items:**
- **P2-a.** FSDP2 has no `MixedPrecisionPolicy`, so it all-gathers and reduce-scatters in fp32 (`fsdp.py:82-91`). verl's default is bf16 params with fp32 reduce. Porting it costs the "fp32 LM head" guarantee unless the head stays unsharded or keeps fp32 (`hf.py:94-95`).
- **P2-b.** GRPO std is population std + 1e-6. TRL, OpenRLHF and verl use unbiased std + 1e-4 or 1e-9 (`objectives.py:107`).
- **P2-c.** `reinforce_pp` ≠ OpenRLHF REINFORCE++ (§1).
- **P2-d.** No sequence-product IS correction. TRL's default is `sequence_mask`.
- **P2-e.** `const_max_len` uses the batch max, not L_max (`objectives.py:60`, `trainer.py:638`).
- **P2-f.** weightsync mixes global and group ranks (`weightsync.py:175,189-193,274,281`). **[ran]** A group `[1,2,3]` fails with "Global rank 0 is not part of group".
- **P2-g.** `ranks=` creates a new process group per update (`:227`), which leaks one NCCL communicator per call.
- **P2-h.** Receivers clone the whole model into device memory before commit (`:155-157`).
- **P2-i.** The 2×bucket buffer (up to 16 GiB) is left in the caching allocator (`:244`).
- **P2-j.** The owner stages from pageable CPU memory: `FSDPTrainer.state_dict` goes to CPU (`fsdp.py:170`), then H2D again.
- **P2-k.** `_reduce_grads` flattens only non-`None` grads (`trainer.py:712`). MoE experts unused on one replica give different bucket sizes, and the step hangs (DDP-replica path only).
- **P2-l.** The 30B-A3B refusal on one node is honest, but no CPU-offload plan exists, although `FSDPTrainer(offload_policy=CPUOffloadPolicy())` is accepted (`fsdp.py:83-84`).

## 1. Algorithm parity, run against verbatim TRL and OpenRLHF code (`parity.py`)

- **Setup:** B=8, T=12, unequal lengths (12, 3, 7, 12, 1, 9, 5, 11), float64.
- **Method:** each row compares the loss and the autograd gradient w.r.t. per-token logp.
- **Result:** every listed equivalence is exact (0.0) or at machine epsilon.

| TRL 72cf37d (`T:3012-3111`) / OpenRLHF | Crucible | loss diff | grad max diff | Verdict |
|---|---|---|---|---|
| `dapo` (default), eps 0.2/0.28 | `ppo` + `token_mean_global` | 0 | 0 | equal |
| `bnpo` | `ppo` + `token_mean_global` | 0 | 0 | equal on 1 process |
| `grpo` | `ppo` + `seq_mean_token_mean` | 0 | 0 | equal (but D11 at scale) |
| `dr_grpo`, `max_completion_length=512` | `const_max_len`, `max_len=512` | 0 | 0 | equal **only** with an explicit `max_len`, which the trainer cannot pass. With the default batch max: loss 0.243 vs 0.0057, grad ×41.7 (P2-e) |
| `delta=3` (TRL two-sided clip) | `ppo dual_clip=3` | 0 | 0 | equal |
| `cispo`, `epsilon_high=5` (absolute cap) | `cispo {m2: True, eps_high: 4}` | 0 | 0 | equal under that mapping. Crucible's default (M1 bounds [0.8, 1.28]) differs: grad rel 0.54 |
| GSPO: `importance_sampling_level="sequence"`, dapo or grpo | `gspo` (GSPO-token form) | 0 | 3.5e-18 | equal: sequence-level gradient = token form with `sg(s)` |
| vLLM IS `token_truncate`, cap 3 | `tis C=3` | 0 | 0 | equal |
| vLLM IS `token_mask` [0.5, 2] | `icepop α=0.5 β=2` | 0 | 0 | equal |
| vLLM IS **`sequence_mask`** (TRL default per `C:930`; the docstring `C:330` says `token_truncate`): exp(**Σ** log diff), mask > 3 | none (`mispo`/`hispo` use geometric means) | — | — | gap P2-d. Synthetic: sequence ratios 0.33-3.11, 1/8 masked |
| TRL advantages `(r−μ)/(std_unbiased+1e-4)` (`T:2688-2739`) | `group_advantages(std_normalize=True)`: population std + 1e-6 | — | — | Crucible larger by **1.414× / 1.155× / 1.069×** at n=2/4/8 (P2-b). OpenRLHF `group_norm` matches TRL within 3e-4 |
| OpenRLHF `rloo` (`EM:268-270`) | `rloo` | — | 6e-8 | equal |
| OpenRLHF `reinforce` (REINFORCE++): token-level global whitening, length-weighted (`EM:324-338`) | `reinforce_pp`: sequence-level whitening | — | **0.326** | differ when lengths differ; equal lengths agree to 2.7e-6 (P2-c) |
| OpenRLHF `reinforce_baseline` (group mean, then global whitening) | none (`mean` has no global whitening) | — | 1.15 (scale ×2.53) | missing estimator (P2-c) |
| OpenRLHF reward clamp ±10 after the baseline (`models/utils.py:131-159`) | none | — | rloo with R=30: 10 vs 30 | missing (P2-c) |
| TRL `use_bias_correction_kl=True` (default): k3 × ratio (`C:967`, `T:3033`); OpenRLHF KL-in-reward k1, β=0.01 default | k3 as a loss, no ratio | — | — | convention gaps, documented |

**Contract notes:**
- TRL's `dapo` normaliser is `num_items_in_batch / num_processes`, with DDP averaging the gradient. That equals Crucible's global token count with `set_gradient_divide_factor(1.0)`. Confirmed at 8 ranks: grad-norm equality in §3.
- TRL's default `epsilon_high` equals `epsilon` (0.2). Crucible defaults to 0.28.

## 2. Weight sync vs checkpoint-engine (PATTERN claim)

| Step | checkpoint-engine 84631a2 | Crucible `weightsync.py` | Gap |
|---|---|---|---|
| Bucket size | `min(max(PS_MAX_BUCKET_SIZE_GB=8, max_tensor), free/3 \| free/2)` over the min free memory of all ranks (`CE:690-719`) | same rule (`:137-146`, `:196-200`) | faithful |
| Source memory | trainer shards registered as **pinned host memory**, with H2D on every rank in parallel through an h2d buffer (`pin_memory.py`, `CE:314-369`) | the owner's `owned[name]` (pageable CPU from `FSDPTrainer.state_dict`) is copied into the device buffer (`:276-280`) | P2-j |
| OOM guard | `dist.barrier` before allocating buffers (`CE:838-839`) | none; allocates 2×B on every update (`:244`) | P2-i |
| Receiver | **CUDA IPC** of the bucket buffer to the vLLM worker through ZMQ, loaded in place (`worker.py:134-214`, `VllmColocateWorkerExtension`) | `StagedState.load` clones every tensor, then `engine.load_weights(dict)` (`:155-161`, `:363-364`) | D7, P2-h |
| Error code | per bucket, `all_reduce(SUM)` of ret codes (`CE:935`) | per bucket, `all_reduce(MAX)` of rank+1 (`:256-267`) | equivalent; Crucible names the culprit |
| End barrier | store-based TCP barrier (`CE:552-573`) | `dist.barrier(group)` (`:296`; on NCCL needs `set_device`) | note |
| P2P / RDMA | mooncake RDMA P2P for elastic joins | portable plan JSON only (`:102-108`) | gap (needed only beyond one cluster) |
| Measured | Qwen3-235B bf16 on 8×H800 TP8: 6.22 s broadcast (README:47-62) | 7.62 GB/s broadcast on H100×4 (`02-live-validation.md` §3) | — |

**8-rank gloo runs [ran]**
- Driver: `run_world.py` / `ws8.py`, reusing `tests/unit/_weightsync_worker.py` unchanged.

| scenario @ W=8 | s | result |
|---|---|---|
| broadcast (3 buckets, 9 dtypes, 2 versions) | 1.5 | pass, all 8 bitwise |
| slow_reload | 1.0 | pass, bitwise |
| sharded owners (0, 1) | 0.9 | pass, bitwise |
| fail (rank 2, 3rd bucket) | 0.9 | every rank raises naming rank 2 / bucket 2; versions stay 1; state unchanged |
| subgroup [0, 2] | 0.9 | pass; non-members return `None` |
| publisher → 7 receivers | 0.9 | pass, versions [1, 2], bitwise |
| dp_exact / dp_broadcast | 2.1 / 2.2 | replicas bitwise; vs full batch: dis 5.2e-7, ppo 3.0e-8; imbalance 1.146 |

**What changes on NCCL at 8 ranks:**
1. `set_device(local)` must come before `init_process_group`. The `dist.barrier` at `:296` otherwise guesses the device.
2. The reload thread and the NCCL stream are ordered only because both use the default stream. A future side stream needs `record_stream` and events.
3. Control tensors already ride `_cdev` (B.23).
4. NCCL timeouts: a publish of 16 GB at 7.6 GB/s takes about 2 s, far under the 10-min watchdog. A disk publish blocks followers in the **gloo** command group, which is correct (`spmd.py:27-34`).
5. `NCCL_CUMEM_ENABLE`: Modal needed `=0` with NCCL 2.30.7 (B.26). OpenRLHF sets `=0` only for vLLM < 0.16 and says it breaks `ncclCommInitRank` on NCCL ≥ 2.27 with vLLM ≥ 0.16 (`trainer/ray/ppo_actor.py:502-509`). Crucible sets nothing in `src`, so test both settings in plan stage 0.

## 3. FSDP2 / SPMD at larger world sizes [ran]

- Driver: `fsdp8.py`, reusing `_fsdp_worker.py` unchanged, plus `_edge_worker.py`.

| scenario | W | s | max \|Δ\| vs single process | notes |
|---|---|---|---|---|
| fsdp step/layer × dis/ppo + skew | 8 | 3.6 | 1.42e-6 | grad norms equal the reference to 6 digits; params and moments per rank = 0.125; at W=8 the skew case pads 0 micro batches, so the "padding ran" assertion is world-dependent |
| hsdp (2, 4) | 8 | 2.1 | 1.42e-6 | params per rank 0.25 |
| hf_layer (Llama, 3 blocks) | 8 | 4.4 | 4.6e-7 | |
| spmd_run (real Runtime over HSDP 2×4) | 8 | 3.0 | restore 0.0 | reward 0.055 → 0.240, staleness ≤ 1, DCP checkpoints at steps 8 and 16, resume at step 8 |
| fsdp + hf_layer | 3, 5, 6, 7 | 2-6 | ≤ 2.5e-6 | odd and uneven shards fine |
| DCP save → load (reshard) | 8→4, 4→8, 8→3 | ~2 | restore 0.0; next step 6.0e-8 | moments restored |
| **few items** | 8 (7 items), 2 (1 item) | timeout | — | **D1** |

**Checks:**
- **Clip on DTensors:** correct, with `gn.full_tensor()` at `trainer.py:651`; shown by the grad-norm equality at W=8.
- **Optimizer moments:** sharded.
- **DCP:** reshards across world sizes.
- **Mixed precision:** none (P2-a).
- **Activation checkpointing:** HF `gradient_checkpointing_enable` plus train mode (`hf.py:30-45`); works under layer mode.
- **Sequence packing:** none (D9).

## 4. 8×H100 readiness: `placement.plan` output [ran]

**Standard 1 node × 8 H100 80 GB**
- Workloads: 1280 seqs of 1k + 2k; agentic 256 × (8k + 8k) with 8 turns × 5 s; calibrated (mfu 0.167, decode curve), 512 × (1k + 2k).

| model | workload | plan | engine | mesh, reshard | step s | production reference (8 GPUs) |
|---|---|---|---|---|---|---|
| Qwen3-4B | 1k/2k × 1280 | disaggregated, 3 train + 5 engines | TP1 × 5, 150 seqs each | (1, 3) layer | 91.5 | verl `run_qwen3_4b_fsdp.sh`: colocated, vLLM TP2 mem 0.6, dynamic batch 3000 / 4096 tokens per GPU, grad ckpt, remove_padding |
| Qwen3-4B | agentic | colocated_partial | TP1 × 8 | (4, 2) layer | 313.8 | SkyRL Megatron DAPO 4B: vLLM 4×TP2 mem 0.8 |
| Qwen3-8B | 1k/2k × 1280 | **colocated_partial** (D3) | TP1 × 8, 131 seqs each | (2, 4) layer | 136.3 | verl `run_qwen3_8b_fsdp.sh`: colocated, vLLM **TP2 mem 0.6**, dynamic 24576 tokens per GPU, grad ckpt, remove_padding; NeMo-RL 8B: TP1 mem 0.6, packing 4096 / 8192 |
| Qwen3-8B | calibrated | elastic, 1 rollout + 7 trainer GPUs | TP1 | (1, 7) | 70.0 | — |
| Qwen3-30B-A3B | any | **PlacementError** | — | — | — | verl FSDP2: TP4 + **param and optimizer offload**; slime: TP8 SGLang + CPU Adam; SkyRL/NeMo-RL: **2 nodes** with EP8 |

**REPACSS shapes: 4 × H100 NVL 94 GB per node**
- Planner settings: 835 dense TFLOPs and the measured 7.62 GB/s publish.

| nodes × GPUs | Qwen3-4B | Qwen3-8B | Qwen3-30B-A3B |
|---|---|---|---|
| 1×4 | disaggregated, 3 train + 1 engine, (1, 3) | colocated TP1 × 4, (1, 4) layer (D3) | PlacementError |
| 2×4 | elastic, (1, 7) **across nodes** (D10) | elastic, (1, 7) across nodes (D10) | colocated TP2 × 4, (1, 8) (**D2**: cannot initialize) |

**Sane layout for 2×4 H100 NVL (recommended, by hand)**
- **Qwen3-8B:** node 0 = 4 FSDP ranks with mesh (1, 4) **inside the node** and reshard layer. Node 1 = 2 vLLM TP2 engines on the NVL-bridged pairs, or 4 × TP1. Disaggregated, `CheckpointPublisher` to node-local NVMe with GC.
- **Qwen3-30B-A3B:** needs D2 fixed, then either 2 nodes of FSDP with `CPUOffloadPolicy` and vLLM TP4, or a third node.

## 5. 8-GPU test plan for REPACSS (2 nodes × 4 H100 NVL)

Fix first: D1, D5, D6 and D8. The unit workers also need `cuda:{RANK % gpus_per_node}`.

| # | Stage | Uses / add | Pass criteria | Wall | GPU-h |
|---|---|---|---|---|---|
| 0 | Allocation smoke | **add** `scripts/repacss_sbatch.sh` (2 nodes, `srun --ntasks-per-node=4`, `RANK = SLURM_PROCID`, `LOCAL_RANK = SLURM_LOCALID`, `MASTER_ADDR` = first host) → `cluster.smoke`; `nvidia-smi topo -m`; a torch NCCL all_reduce bandwidth sweep, intra- and inter-node; `NCCL_CUMEM_ENABLE` 0 vs 1; `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / IB HCA | rank_sum = 28 at W=8; intra-pair, intra-node and inter-node busbw recorded; no init hang under either CUMEM setting | 15 min | 2 |
| 1 | NCCL unit suite | `scripts/modal_infra_check.py` scenarios via srun at W=4 (1 node) and W=8 (2 nodes); **add** the D1 regression (7 items / 8 ranks) | the gloo assertions hold, with Δ ≤ 5e-6 (§3 gives the W=8 gloo numbers); fail, subgroup and publisher pass; spmd_run learns and resumes bitwise | 20 min | 3 |
| 2 | Weight sync at size | `BroadcastWeightSync` on Qwen3-8B bf16 (16.4 GB), rank 0 → 3 and → 7 receivers; `CheckpointPublisher` to node-local NVMe vs Lustre | bitwise; ≥ 7 GB/s intra-node (NVL pairs faster); inter-node ≥ 70% of NCCL busbw; publish under 10 s | 20 min | 3 |
| 3 | FSDP2 8-rank parity, real model | Qwen3-1.7B: `FSDPTrainer` layer mode, mesh (1, 8) vs (2, 4) vs one-GPU `TorchTrainer`, same batch, 2 updates; DCP save at 8 → load at 4; Qwen3-8B (2, 4) peak memory at `token_budget` 8192 and 16384 | relative weight Δ ≤ 1e-4 (bf16 autocast); grad-norm Δ ≤ 1e-3 relative; 8B peak ≤ 85 GB; (2, 4) faster than (1, 8) | 45 min | 6 |
| 4 | vLLM on H100 NVL | `VLLMServerEngine` TP1 and TP2 (bridged pair), `--enable-sleep-mode`; 50 cycles of sleep L1 → wake weights → load checkpoint → wake KV; `probe_logprobs_mode` | memory after wake within 1 GB of the first; logprobs processed; `rollout_probs_diff` vs FSDP trainer under 0.02 on Qwen3-1.7B | 40 min | 3 |
| 5 | Disaggregated GRPO, 1.7B | `gate_dist` via sbatch: node 0 trainer (4), node 1 engines (4); word-file task, 20 steps | held-out success ≥ 0.9 (0.6B went 0.44 → 1.00); staleness ≤ τ; 0 masked; no watchdog; publish ≤ 15 s | 45 min | 6 |
| 6 | Full GRPO, Qwen3-8B (never established) | same as stage 5 with 8B on the backlog task, 60 steps, `wall_clock` ≥ 4 h, checkpoint every 10 steps | held-out improves with a CI excluding 0; no D1/D3/D8 failure; restart from checkpoint works | 3-4 h | 24-32 |

**Total:** about 50-55 GPU-hours.

Colocated FSDP + vLLM (D3) stays out of this plan until `FSDPTrainer` gets a real offload: `CPUOffloadPolicy` at the barrier, or `empty_cache` plus shard offload.

## 6. Picks

| Pick | From | Verdict | Effort | Why |
|---|---|---|---|---|
| Fixed metric-key all-reduce (or all-gather the key set) | own fix | — | S | D1 |
| Meta-device init + `set_model_state_dict(..., broadcast_from_rank0=True)` | torch DCP API; verl and NeMo-RL pattern | PATTERN | S-M | D2 |
| `VllmColocateWorkerExtension` + `ParameterServer.update` (bucketed broadcast + CUDA IPC into vLLM) | checkpoint-engine (MIT) | **DEPEND** | M | D7: the vLLM receiver Crucible lacks; keep Crucible's `weightsync` for TorchEngine and tests |
| NCCL group with vLLM workers (`StatelessProcessGroup` + `PyNcclCommunicator`, world 1 + engines×TP) and the CUDA IPC colocate path | OpenRLHF `trainer/ray/ppo_actor.py:101-151,409-490`, `vllm_worker_wrap.py:32-70` | PATTERN | M | the alternative to DEPEND for D7 |
| TRL `rollout_func` bridge fix: no inner G loop, keep masked rows aligned, `lp[0]`, one batched generate per call | TRL (Apache-2.0) | DEPEND (fix bridge) | S | D4 |
| Sequence-product IS modes (`sequence_mask` / `sequence_truncate`) | TRL `T:2589-2636` | PORT | S | P2-d |
| REINFORCE++ token whitening, `reinforce_baseline`, reward clip, KL-in-reward | OpenRLHF `EM:237-340`, `models/utils.py:131-159` | PORT | S | P2-c |
| Unbiased-std option + eps | TRL / verl convention | PORT | S | P2-b |
| `remove_padding` / varlen packing + token-balanced micro batches | verl, NeMo-RL | PATTERN | M | D9 |
| FSDP2 `MixedPrecisionPolicy(param bf16, reduce fp32)`, LM head kept fp32 | verl `transformer_impl.py:395-405` | PATTERN | S | P2-a |
| Ray-on-Slurm sbatch (head + workers, `srun --overlap` driver) or a plain srun launcher | verl `ray_on_slurm.slurm` (fix its `nodes_array=("$nodes")` quoting bug, :47); NeMo-RL `ray.sub` | PATTERN | S-M | D5 |
| Node-local shard groups + communication term in `train_s` | own fix | — | S | D10 |
| Checkpoint GC (keep last k) | own fix | — | S | D8 |
