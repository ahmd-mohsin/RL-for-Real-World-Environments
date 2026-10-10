# M12 research 01: verl and slime, the RL-infrastructure layer

Date: 2026-10-09. Scope: ARCHITECTURE §4 M12 `train` (RL-infrastructure subsection), DESIGN §10.3 (trainer-side contracts 1–8) and §10.4 (long horizons), `docs/research/2026-10-09-agentic-rl-infra-survey.md`, `src/crucible/train/rollout.py` (`crucible.rollout/v1`).

**Method.** This is a read-only survey of three pinned checkouts (`00-source-pins.txt`). Four parallel surveys covered:
- verl topology, training, weights and checkpoints;
- verl rollout, async, agentic and mismatch, plus the MiMo diff;
- slime topology, training, weights and checkpoints;
- slime async, agentic, mismatch and balance.

I then re-read every load-bearing claim myself; those are marked ✓. Other line references come from the surveys, at the pinned commits. Where two surveys disagreed I resolved it by grep (e.g. slime `--log-probs-max-tokens-per-gpu` is dead, §S2.4). Nothing was modified, committed, or fetched over the GitHub API.

**Path prefixes**

| Prefix | Path (under `RL-Environments/06-frameworks-and-env-code/`) | Commit | License |
|---|---|---|---|
| `V/` | `infra-volcengine-verl/` | 5eb4826 (2026-10-10) | Apache-2.0 (`Notice.txt`: "Copyright 2023-2024 Bytedance Ltd.") |
| `MV/` | `xiaomimimo-verl/` | a2ad9f6 (2026-09-26) | Apache-2.0 |
| `S/` | `infra-thudm-slime/` | c933787 (2026-10-09) | Apache-2.0 (no NOTICE file; `S/slime/utils/seqlen_balancing.py:1` is copied from verl and keeps the Bytedance header) |

Short forms: `VY` = `V/verl/trainer/config/_generated_ppo_trainer.yaml` (the flattened config), `RY` = `V/verl/trainer/config/rollout/rollout.yaml`, `SA` = `S/slime/utils/arguments.py`.

---

## 0. Executive summary

1. **verl at this pin is no longer the verl most write-ups describe.**
   - The default trainer is **v1** (`trainer.use_v1: true`, VY:982 ✓). The old `RayPPOTrainer` is deprecated, "will be removed in v0.9.0" (`V/verl/trainer/main_ppo.py:183-191`).
   - v1 has three modes: `sync`, `colocate_async` and `separate_async` (VY:984 ✓).
   - Data lives in **TransferQueue**, a KV store forced on by v1 (`main_ppo.py:137-143` ✓).
   - A **ReplayBuffer** applies the staleness cap, DAPO filtering and failed-group refill.
   - `fsdp_workers.py` and `megatron_workers.py` are gone; every role runs on a pluggable engine (FSDP/FSDP2, Megatron, VeOmni, TorchTitan, AutoModel).
2. **`colocate_async` is exactly the topology DESIGN §10.3 point 2 chose for the first M12 run:** colocated, async, partial rollout.
   - At sample end it aborts in-flight requests and sleeps the engine. At step end it updates weights and resumes (`V/verl/trainer/ppo/v1/trainer_colocate_async.py:37-48` ✓).
   - The client re-submits prompt + partial tokens under the new weights, at the cost of a full re-prefill (`V/verl/workers/rollout/llm_server.py:225-361` ✓).
   - Staleness is capped by `max_off_policy_threshold: 8` with `drop` | `wait` (VY:1001-1002 ✓).
   - slime has **no** equivalent on one GPU. Its in-place (PipelineRL-style) async is refused with `--colocate` (SA:2248-2253). Its "fully async" mode needs separate GPUs in practice and has **no staleness cap at all** (`S/docs/en/advanced/rollout-scheduling.md:76`).
3. **Neither framework records the policy version per span, which is what Crucible's record requires.**
   - verl keeps `min/max_global_steps` per trajectory (`llm_server.py:333-337`, `:357-359` ✓) and measures age from **prompt submission** (`V/verl/trainer/ppo/v1/replay_buffer.py:520-529` ✓).
   - slime appends one version per *finished SGLang request* (`S/slime/utils/types.py:487-488` ✓). Its adapter-based agent samples carry none.
   - `crucible.rollout/v1` `Turn.policy_version` is strictly richer than both. M12 must fill it itself, at each resume boundary.
4. **Mismatch tooling is mature in both, and off by default in both.**
   - verl: `rollout_is: null` by default (`V/verl/trainer/config/algorithm/rollout_correction.yaml:6` ✓). It offers TIS / IcePop / k1–k3 rejection sampling, including a geometric-mean filter, plus a bypass mode.
   - slime: `--use-tis` is False by default. It offers TIS, IcePop, an MIS example menu and OPSM.
   - Both have R2/R3 MoE routing replay.
   - Neither has KPop (binary-KL), a routing-confidence gate, or a general FP32-LM-head switch. verl has `lm_head_precision` only in the AutoModel engine.
   - verl logs `rollout_probs_diff_*` (`V/verl/utils/debug/metrics.py:76-79` ✓). slime logs `train_rollout_logprob_abs_diff`.
5. **Critic mode:** verl's GAE already **skips observation tokens** in the Bellman chain (`V/verl/trainer/ppo/core_algos.py:254-256` ✓), which is SAO's skip-observation GAE. slime's batched GAE takes no mask (`S/slime/utils/ppo_utils.py:648-721` ✓). This matters for Crucible's single-feedback critic mode.
6. **Recommendation:**
   - **DEPEND on verl v1 (`colocate_async`, FSDP2 engine, vLLM TP=1) for the M12 lab-5090 gate.** Crucible plugs in as a registered agent loop or advantage estimator. It does not fork the trainer.
   - **PORT** the small, CPU-testable pieces:
     - verl `seqlen_balancing.py`, `rollout_corr_helper.py`, `debug/metrics.py` and `router.py`;
     - slime `dp_schedule.py`, `weight_sync.should_flush_cache`, `staleness.py` and the `mis.py` menu;
     - MiMo's infra-aware DAPO classifier.
   - **Do not DEPEND on slime** for the MVP. It is Megatron-only, runs on about 5.8k lines of SGLang/Megatron patches, and assumes 8-GPU nodes. Take slime's disaggregated, fault-tolerance and TrajectoryManager designs as PATTERN.

---

# Part V: verl (upstream, `V/`)

## V0. Which code you are reading

- **Two trainers coexist.**
  - **v1:** `V/verl/trainer/ppo/v1/`, a `PPOTrainer` base plus three registered modes (`v1/__init__.py:15-30`; `trainer_sync.py:24`, `trainer_colocate_async.py:19`, `trainer_separate_async.py:46`).
  - **v0:** `RayPPOTrainer`, `V/verl/trainer/ppo/ray_trainer.py:286`, deprecated.
  - The legacy experimental recipes `V/verl/experimental/{one_step_off_policy,fully_async_policy}` still exist on a `SeparateRayPPOTrainer` base (`V/verl/experimental/separation/ray_trainer.py:52-57`). `separate_async` describes itself as "v0 fully-async semantics" when `hybrid_engine=False` (`trainer_separate_async.py:78-81`).
- **Unified engine workers.**
  - `TrainingWorker` (`V/verl/workers/engine_workers.py:78`) and `ActorRolloutRefWorker` ("Hybrid worker that includes actor model, rollout and optional ref model"; "no longer support spmd mode and run native server mode", `:453-457`).
  - Backends register via `EngineRegistry` (`V/verl/workers/engine/base.py:339`).
  - The engine API is `train_batch` = zero_grad → `forward_backward_batch` → `optimizer_step` (`base.py:113-131`); `infer_batch` = no-grad forward (`:134-149`); `get_per_tensor_param()` yields HF-named full tensors for weight sync (`:151-160`).
  - `grep -rn "fsdp_workers\|megatron_workers" verl` returns nothing.
- **Rollout is always server-mode async:** "mode is always "async" since sync mode is deprecated" (`ray_trainer.py:921-923`).

## V1. Topology and placement

**Ray pools and placement groups**
- `ResourcePoolManager` maps pool → `[gpus_per_node]*nnodes` and role → pool. `max_colocate_count: int = 3` (`V/verl/single_controller/ray/base.py:196` ✓).
- `get_placement_groups(strategy="STRICT_PACK")` creates one PG **per node**, one bundle `{"CPU": max_colocate_count, GPU: 1}` per GPU (`base.py:135-166`). Each worker asks for `num_gpus = 1/max_colocate_count` (`base.py:637` ✓), so up to three worker groups can share a GPU bundle.
- Pools (`V/verl/trainer/ppo/v1/trainer_base.py:788-848`):
  - `global_pool` for actor + rollout (+ ref) and critic;
  - an optional `reward_pool` (`reward.reward_model.enable_resource_pool`, `:823-831`), otherwise the RM maps to `global_pool`;
  - an optional `teacher_pool`.
  - Standalone rollout replicas create their own per-replica pool with `max_colocate_count=2` (`V/verl/workers/rollout/replica.py:194-212`).
- **All roles in one pool are fused into one Ray process per GPU.** `create_colocated_worker_cls` builds a `WorkerDict` that instantiates every role in-process (`single_controller/ray/base.py:1008-1048`). Actor, ref and critic therefore share one process, and one OOM kills all three. Per-role pools with different parallel sizes need custom code (comment, `ray_trainer.py:846-849`).
- **Ref** is fused into `ActorRolloutRefWorker`. With LoRA, ref = actor with the adapter disabled (`main_ppo_v0.py:43-53`, `ray_trainer.py:1271-1277`).

**Rollout placement modes** (`replica.py:49-62`):
- `HYBRID`: shares GPUs with the trainer and switches context with a weight sync (on-policy).
- `COLOCATED`: same PG, separate process, no weight sync (used for GRM / LLM-judge reward models).
- `STANDALONE`: own GPU pool (disaggregated).

**What "hybrid" actually is on vLLM**
- The trainer process holds only a thin `ServerAdapter` (`engine_workers.py:656-680`).
- `vLLMReplica.launch_servers` starts one `vLLMHttpServer` Ray actor per node per replica, with node affinity and **no GPU reservation**. It sets `CUDA_VISIBLE_DEVICES` to the trainer workers' GPUs and launches AsyncLLM with `distributed_executor_backend: "mp"` (`V/verl/workers/rollout/vllm_rollout/vllm_async_server.py:138`, `:332`, `:1494-1556`).
- Net per GPU: **1 trainer process (all fused roles) + 1 vLLM worker process**, +1 if a COLOCATED GRM is used. Add one CPU-side HTTP-server actor per node per replica.
- Replicas = `trainer_world_size // (rollout TP·DP·PP)` (`llm_server.py:444-470`). The trainer world size must be divisible by the rollout world size (`engine_workers.py:661-667`).
- Initial rollout weights are `load_format: dummy` (VY:323); real weights arrive at the first `update_weights`.

**Phase switching: offload, sleep/wake, KV release**
- **Sync step:** sample → `on_sample_end` sleeps replicas ("discard weights and kv cache", `trainer_sync.py:40-43`) → reward / old-logprob / ref / values / advantage / critic / actor (`trainer_base.py:562-614`) → `on_step_end` `update_weights` wakes them (`trainer_sync.py:35-38`).
- **vLLM sleep levels:** level 2 (discard weights + KV), or level 1 for LoRA, MTP drafter or NPU (`vllm_async_server.py:1432-1446`).
  - It calls `engine.sleep()` rather than `collective_rpc` so that all DP shards free memory; otherwise you get "OOM during FSDP training backward when DP > 1" (`:1468-1478`).
  - Wake re-allocates `["kv_cache","weights"]` and resets the prefix cache (`:873-892`).
  - Gated by `free_cache_engine: true` (VY:310); `enable_sleep_mode: True` (`V/verl/workers/config/rollout.py:274`).
- **SGLang:** `enable_memory_saver: True` and `release/resume_memory_occupation` (`V/verl/workers/rollout/sglang_rollout/async_sglang_server.py:316`, `:448-511`).
- **Colocated weight-sync memory order** (`ActorRolloutRefWorker.update_weights`, `engine_workers.py:735-837`):
  1. disable expandable segments + `empty_cache`;
  2. `rollout.resume(tags=["weights"])`;
  3. stream `get_per_tensor_param()`;
  4. offload trainer params if `param_offload`;
  5. `synchronize` + `empty_cache`;
  6. **barrier across all trainer ranks** (`:826-831`);
  7. `rollout.resume(tags=["kv_cache"])`.
  Weights come back first and KV last, which minimizes peak memory.
- **Trainer offload:** `param_offload` / `optimizer_offload`, both default `False` (`V/verl/workers/config/engine.py:90-92`).
  - `train_mode()/eval_mode()` contexts load and offload automatically (`V/verl/workers/engine/base.py:300-337`). Eval loads params only; train loads params, optimizer and grads.
  - `train_mini_batch` suspends auto-offload inside the mini-batch loop (`engine_workers.py:283`, `:306-311`).
  - FSDP1 actor `CPUOffload` is **forced off**, "because it causes incorrect results when using grad accumulation". Ref gets `CPUOffload(offload_params=True)`; FSDP2 uses `CPUOffloadPolicy(pin_memory=True)` (`V/verl/workers/engine/fsdp/transformer_impl.py:425-466`).
  - Megatron offload reuses one pinned CPU buffer per DDP buffer (`V/verl/utils/megatron_utils.py:460-490`).
- **Rollout memory:** `gpu_memory_utilization: 0.5` (VY:305) with a `standalone_gpu_memory_utilization` override. Rollout is created last so vLLM can size the KV cache (`ray_trainer.py:892-894`).

**Disaggregated / mixed**
- **`separate_async`** (`V/verl/trainer/ppo/v1/trainer_separate_async.py`):
  - a standalone pool `rollout.nnodes × rollout.n_gpus_per_node` (> 0 asserted, `:65-68`), which needs a non-`naive` checkpoint engine (`:69-71`);
  - colocated RM rejected (`:72-77`);
  - trainer GPUs can be **lent to rollout** between steps under a cost/benefit rule: switch if `remaining × per_sample_time × (1 − 1/scaling_factor)` exceeds the measured switch cost (`:318-339`), with an adaptive reclaim threshold (`:290-306`);
  - `DetachActorWorker` keeps CPU copies of the actor so every mini-batch in a `parameter_sync_step` cycle computes π_old with the same weights (`:188-212`; `V/verl/experimental/separation/engine_workers.py:36-169`).
- **PD disaggregation:** `rollout.disaggregation.*` (VY:429-436; footprint `llm_server.py:449-464`).
- **Reported numbers** (doc only, not reproduced): Qwen3.5-35B-A3B, 4 nodes, 32k responses.

  | Mode | Wall time | Throughput |
  |---|---|---|
  | `sync` | 22.79 h | 12,053 tok/s |
  | `colocate_async` | 14.72 h (−35.4%) | 18,852 tok/s |
  | `separate_async` | 14.10 h (−38.1%) | — |

  Source: `V/docs/advance/v1_async_trainer.md:276-289`. Switch on vs off: −10.3% (`:302-319`).

## V2. Training parallelism

**Backends**
- **FSDP/FSDP2:** selected by `strategy`.
  - Mesh from `fsdp_size` (`-1` = full world, VY:35).
  - `reshard_after_forward: true` = ZeRO-3 (VY:34).
  - Mixed precision: params bf16, reduce fp32 (`fsdp/transformer_impl.py:395-405`).
- **Ulysses SP** (FSDP engine): `(dp, sp)` mesh, requires remove-padding (`fsdp/transformer_impl.py:148-152`, `:248-256`). `actor.ulysses_sequence_parallel_size` is deprecated and forwarded into `fsdp_config` (`V/verl/workers/config/actor.py:294`, `:325-326`).
- **Megatron** (`McoreEngineConfig`, `V/verl/workers/config/engine.py:184-209`). Defaults:

  | Key | Default |
  |---|---|
  | TP | 1 |
  | EP | 1 |
  | ETP | None |
  | PP | 1 |
  | VPP | None |
  | CP | 1 |
  | `dynamic_context_parallel` | False |
  | `sequence_parallel` | True |
  | `use_distributed_optimizer` | True |
  | `use_mbridge` | True |

  DP is implicit, `n_gpus/(TP·PP·CP)` (`V/verl/utils/config.py:141-152`). Dynamic CP schedules over the DP×CP plane (`megatron/transformer_impl.py:828-852`) and is incompatible with VPP>1.

**Batch vocabulary**

| Term | Unit | Default | Cite |
|---|---|---|---|
| `data.train_batch_size` | prompts per step | 1024 | VY:509 |
| `rollout.n` | responses per prompt | 1 | VY:331 |
| `actor.ppo_mini_batch_size` | **prompts** per optimizer step (trainer multiplies by n) | 256 | VY:80 ✓; `trainer_base.py:1829-1830` |
| per-GPU mini-batch | `mini×n // dp` (divisibility asserted) | — | `engine_workers.py:266-270` |
| micro-batch | `ppo_micro_batch_size_per_gpu`, or token-budget bins | null | VY:82 ✓ |
| `ppo_epochs` | passes | 1 | VY:111 |
| `parameter_sync_step` | local updates per weight sync | 4 (separate_async) | VY:990 ✓ |

- Optimizer steps per step = `train_batch_size/ppo_mini_batch_size × ppo_epochs`.
- `separate_async` requires `train_batch_size == parameter_sync_step × ppo_mini_batch_size` (`trainer_separate_async.py:56-64`).
- **No-op padding:** v1 upsamples the batch with synthetic no-op samples (`is_padding`, zero reward) to a multiple of `lcm(dp, actor_mini·n, critic_mini·n)` (`trainer_base.py:1588-1621`; `V/verl/trainer/ppo/padding_utils.py:151-175`). Dropped or masked groups therefore never break divisibility.

**Dynamic batching** (`rearrange_micro_batches`, `V/verl/utils/seqlen_balancing.py:348-505`)
- Budget `ppo_max_token_len_per_gpu: 16384` (VY:84 ✓), × SP size.
- The micro-batch count `ceil(total/max)` is **all-reduced MAX across DP** (`:398-406`) and rounded to the VPP group (`:407-408`).
- Partitioning is Karmarkar–Karp on workload `24576·L + L²`, an attention-aware proxy calibrated on 7B (`:27-47`).
- Bins are sorted and interleaved so small ones sit at both ends, cutting PP bubbles (`:475-498`).
- **Default drift:** `actor.use_dynamic_bsz: false` in the yaml (VY:83 ✓) but `EngineConfig.use_dynamic_bsz: bool = True` in the dataclass (`engine.py:100` ✓). See pitfalls.

**Packing / remove-padding**
- Batches are converted to nested (jagged) tensors (`ray_trainer.py:1255-1258`).
- The FSDP engine packs each micro-batch into one `(1, total_nnz)` row with `cu_seqlens` (varlen flash attention) (`fsdp/transformer_impl.py:1155-1253`).
- Megatron uses `pad_bshd_to_minibatch_max: true` so the cuDNN plan is built once (`megatron/transformer_impl.py:812-819`).

**Grad accumulation and loss normalization**
- One backward per micro-batch, one optimizer step per mini-batch.
- FSDP grad sync is deferred to the last micro-batch when `use_no_sync_for_gradient_accumulation` (`fsdp/transformer_impl.py:696-726`). Dataclass default `True` (`engine.py:288`) vs yaml `false` (VY:46).
- **Global-token normalization:** `batch_num_tokens` = all-reduce SUM of `loss_mask` over DP for the whole mini-batch (`fsdp/transformer_impl.py:732-739`).
  - `agg_loss` `token-mean` = `masked_sum/batch_num_tokens × dp_size` (`core_algos.py:1170-1175`), so the loss does not depend on micro-batch count or DP size.
  - Also available: `token-sum`, `seq-mean-token-sum(-norm)`, `seq-mean-token-mean` (`:1176-1205`). Default `loss_agg_mode: token-mean` (VY:101).

**Log-prob recompute**
- `compute_log_prob` uses `infer_batch` with `rollout.log_prob_{micro_batch_size_per_gpu,max_token_len_per_gpu,use_dynamic_bsz}`, which interpolate from the actor (VY:326-328; `engine_workers.py:621-629`).
- It is computed **once per data batch** before the mini-batch loop, and skipped in rollout-correction **bypass mode** (`trainer_base.py:1638-1648` ✓).
- MoE `routed_experts` come back from the same pass (`ray_trainer.py:1311-1317`).

**Dispatch:** v1 passes `KVBatchMeta` keys into TransferQueue, and workers resolve them to TensorDicts (`V/verl/utils/transferqueue_utils.py:347-380`). Pinned `TransferQueue==0.1.11` (`V/requirements.txt:28`).

## V3. Rollout parallelism

- **Backends:** vLLM, SGLang, TRT-LLM (`replica.py:385-387`). A replica's world size = TP·DP·PP (`replica.py:102-112`).
- **Defaults (RY):**

  | Key | Default | RY line |
  |---|---|---|
  | `tensor_model_parallel_size` | 2 | :67 |
  | `data_parallel_size` | 1 | :70 |
  | `expert_parallel_size` | 1 (= TP·DP when >1) | :72-75 |
  | `pipeline_model_parallel_size` | 1 | :78 |
  | `max_num_batched_tokens` | 8192 | :81 |
  | `max_num_seqs` | 256 | :87 |
  | `enable_chunked_prefill` | True | :95 |
  | `enable_prefix_caching` | True | :98 |
  | `scheduling_policy` | fcfs | :104 |
  | `calculate_log_probs` | True | :245 |
  | `skip_tokenizer_init` | True (token-in-token-out) | :345 |
  | `logprobs_mode` | processed_logprobs | :101 |

- **SGLang mapping:** `tp_size=TP·DP`, `dp_size=DP`, `ep_size=EP`, `max_running_requests=max_num_seqs` (`async_sglang_server.py:310-337`).
- **Router** (`V/verl/workers/rollout/router.py:133-298`): `GlobalRequestLoadBalancer` is **one Ray actor shared by all agent-loop workers**. Dispatch is **per request** (one model turn), not per batch or group.
  - **Sticky routing:** an LRU of `request_id→server`, size 10000 (`:29`, `:167-187`), keeps a trajectory's turns on one replica for prefix-cache reuse.
  - **Otherwise least in-flight**, with a random tie-break (`:199-201` ✓).
  - **"Full determinism"** uses Python `hash(request_id)` (`:197` ✓). That is salted per process (pitfall).
  - **Elastic** add/remove of servers (`:227-253`) and a pluggable `router_class` (`:360-433`).
  - **vLLM DP inside a replica:** pinned by `crc32(session_id) % dp_size`, load-blind (`vllm_async_server.py:797-802`).
  - The sgl-router is used **only** for reward models (`V/verl/experimental/reward_loop/router/`).
- **Concurrency:**
  - engine `max_num_seqs: 256`;
  - `ray_actor_max_concurrency: 1024`, so control RPCs are not starved (RY:89-92);
  - `agent.num_workers: 8` (RY:258);
  - **no agent-level semaphore**: every sample becomes an asyncio task at once (`V/verl/experimental/agent_loop/agent_loop.py:659-671`; `V/verl/trainer/ppo/v1/agent_loop_tq.py:89-108`). Back-pressure is the engine queue only.
- **Prefix cache** is reset at wake-up, at abort (`clear_cache=True`) and at KV resume (`vllm_async_server.py:877-890`, `:1135-1138`, `:926-933`).

## V4. Synchronous vs asynchronous

| Mode | Where | Topology | Partial rollout | Staleness |
|---|---|---|---|---|
| v1 `sync` | `trainer_sync.py:24-44` | hybrid | no | on-policy |
| v1 `colocate_async` | `trainer_colocate_async.py:19-50` ✓ | hybrid, time-shared | **yes** | `max_off_policy_threshold` (8) |
| v1 `separate_async` | `trainer_separate_async.py:45-446` | standalone + optional lending | yes | same + `parameter_sync_step` (4) |
| one-step-off (legacy) | `V/verl/experimental/one_step_off_policy/ray_trainer.py:265-403` | disaggregated, NCCL | no | exactly 1 |
| fully-async (legacy) | `V/verl/experimental/fully_async_policy/` | disaggregated | yes | `staleness_threshold: 0.1` (sample-count) |

**v1 control flow (all modes)**
1. **The trainer submits prompts.** `prepare_step()` submits one `train_batch_size` per step (`trainer_base.py:1525-1527`). Async modes add `num_warmup_batches` (default 1, VY:987) at `on_train_begin`. Prompts are registered in TransferQueue as `{"is_prompt", "status": "pending", "global_steps"}` (`:1496-1512`).
2. **`AgentLoopWorkerTQ` runs the sessions.** Per prompt it spawns `n` session tasks; the prompt is `finished` only when every session settles, or `failure` if any raised (`agent_loop_tq.py:110-151`).
3. **Sessions write outputs.** Each session writes its output(s) as keys `{uid}_{session}_{index}` tagged with `status, prompt_len, response_len, global_steps, min_global_steps, max_global_steps` (`:180-230`). A session may return **several outputs**, e.g. one per context segment (`:153-158`).
4. **The ReplayBuffer selects.** It polls TQ metadata every 2.0 s (`replay_buffer.py:144`, `:408-419`). Groups are atomic and sampleable only when terminal (`:85-92`). Selection is **oldest submission step first** (`:383-392`).
5. **Training runs.** `step()` runs `parameter_sync_step` mini-iterations: reward → balance → old-logprob → ref → values → advantage → critic → actor (`trainer_base.py:533-612`).

**Queue sizes.** TQ partitions `train`/`val`, and the ReplayBuffer is a view over it, not a bounded queue. In-flight work ≈ `(1+num_warmup_batches)×train_batch_size` prompts plus refills. Sync DAPO caps in-flight work at `max_inflight_gen_batches×train_batch_size` (`replay_buffer.py:446-447`; default 1).

**`colocate_async` hooks** ✓
- `on_sample_end`: `abort_replicas()` then `sleep_replicas()`.
- `on_step_end`: `update_weights(global_steps)` then `resume_generation_replicas()`.
- It uses `FullyAsyncLLMServerClient`, so aborts are invisible to the agent loop.

**Staleness**
- Config: `sampler.max_off_policy_threshold: 8` (null disables), `max_off_policy_strategy: drop`, `sync_refill_failed_groups: false` (VY:1001-1003 ✓).
- Span = `global_steps − prompt_global_steps + 1`, measured from **submission** (`replay_buffer.py:520-529` ✓).
- `drop` evicts terminal stale groups and refills 1:1 (`:576-584`), and asserts that no stale group was selected (`:600-611`).
- `wait` blocks sampling while an in-flight prompt has span ≥ threshold (`:548-556`).
- Both are no-ops in `sync` (`:94-107`).

**Version tracking**
- The server stamps each response with its `global_steps` (`vllm_async_server.py:713`; set during weight updates, `vllm_rollout.py:250,276,319`).
- The resume loop keeps `min_global_steps` from the first chunk and `max_global_steps` from the last (`llm_server.py:333-337`, `:357-359` ✓). Multi-turn loops keep the max across turns (`tool_agent_loop.py:261-267`).
- No per-token or per-span list exists.
- Metrics: `training/off_policy/trajectory_spans/*`, `trajectory_staleness/*` (`trainer_base.py:1970-1990`), and eviction metrics (`replay_buffer.py:359-371`).

**In-flight requests at a weight update: abort, keep the partial tokens, re-prefill under the new weights**
- vLLM:
  1. close the admission gate;
  2. wait for in-flight admissions;
  3. `pause_generation(wait_for_inflight_requests=False, clear_cache=True)` (`vllm_async_server.py:1105-1141`). Aborted requests return their tokens so far with `stop_reason="aborted"` (`:713-756`).
  4. Late arrivals **park** until resume, or are rejected when the replica leaves rotation (`:1165-1187`).
- SGLang: `pause_generation(mode="abort")` (`async_sglang_server.py:702-705`). It cannot reject requests (`:862-877`).
- The client loop re-submits `prompt_ids + tokens_so_far` and concatenates token ids, logprobs, routed experts (new rows only) and top-k. It pins a cumulative `max_tokens` and sleeps 1 s between attempts (`llm_server.py:225-361` ✓). Prefix cache is cleared, so **every resume pays a full re-prefill**.
- The disaggregated sequence is abort → release KV (keep weights) → build group → transfer → finalize → resume KV → resume generation (`V/verl/checkpoint_engine/base.py:509-562`). Generation on synced replicas **stops** for the whole sync.

**Legacy fully-async**
- `MessageQueue` = `deque(maxlen=max_queue_size)` (`message_queue.py:32-37`). When full it **drops the oldest** sample (`:66-83`).
- `max_queue_size = required_samples×(staleness_threshold+1)×trigger_parameter_sync_step` (`fully_async_rollouter.py:491-507`).
- It forces `bypass_mode: True` and `use_rollout_log_probs: True` (`fully_async_ppo_trainer.yaml`).

## V5. Weight synchronization

**CheckpointEngine** (`V/verl/checkpoint_engine/README.md:1-24`; `base.py:49-113`, `:380-562`): `send_weights` / `receive_weights` generators. In rollout, a `CheckpointEngineWorker` on the same GPU as each inference worker process hands tensors over **CUDA IPC**.

| Backend | Transport | Use |
|---|---|---|
| `naive` (default, VY:373 ✓) | in-process generator → ServerAdapter → CUDA IPC over ZMQ (shm fallback) | colocated |
| `nccl` | all-gather to actor rank 0 + NCCL broadcast (`ray.util.collective`) | disaggregated |
| `nixl`, `mooncake` | ring P2P | elastic / heterogeneous |
| `kimi_ckpt_engine` | CPU offload, Mooncake P2P, broadcast | |
| `delta_sharded` | NCCL, changed bytes only | large models |

- **Resharding = full materialization.** FSDP calls `DTensor.full_tensor()` lazily per parameter (`fsdp/transformer_impl.py:1031-1041`); Megatron uses `mbridge.export_hf_weights` (`megatron/transformer_impl.py:981-1009`). The inference engine reshards to its own TP via `load_weights()` (`vllm_rollout/utils.py:414-424`). No layout-to-layout map exists.
- **Buckets:** `update_weights_bucket_megabytes: 2048` (VY:374 ✓). Large tensors are split and re-merged (`base.py:565-633`).
  - NCCL uses two GPU buffers, "device memory overhead is 2 * bucket_size" (`nccl_checkpoint_engine.py:124-125`).
  - Metadata travels over ZMQ PUB/SUB; double-buffered pipelining applies with a single sender (`:95-106`, `:339-371`).
  - `multi_sender=True` adds NVLink-local relays (`:130-142`).
  - Buffers are freed and the group prepared on **every** sync (`base.py:536-537`; `nccl_checkpoint_engine.py:182-193`).
- **FP8:** `rollout.quantization: fp8` injects block-FP8 (e4m3, [128,128]) and bf16 weights are quantized **on the fly at load** (`vllm_async_server.py:1381-1409`; `vllm_rollout/utils.py:400-413`). QAT export happens on the trainer side.
- **Delta sync:** byte-diff against a pinned-CPU snapshot, broadcasting changed `(index,value)` pairs (`delta_checkpoint_engine.py:14-46`). The changed ratio is "~1-3% of parameter bytes for dense models" (`V/docs/advance/delta_weight_sync.md:160-163`).
- **Timing:** colocated `naive` runs once per step after the actor update, with no overlap because rollout is asleep (`trainer_sync.py:35-38`). The policy version passed is `global_steps` (`base.py:354-361`).
- **Stated costs** (docs, not reproduced):
  - Qwen3-30B-A3B to 30 receivers: NCCL ≈ 7 s (8.25 GB/s), Mooncake 5.93 s (`V/verl/checkpoint_engine/README.md:83-89`).
  - Delta vs NCCL (H100): 30B-A3B 7.1 vs 32.2 s; 235B-A22B 11.4–14.9 vs 246–266 s (`delta_weight_sync.md:150-157`).

## V6. Agentic / multi-turn

- **Agent loop:** `AgentLoopBase.run(...) -> AgentLoopOutput` is one coroutine per trajectory (`agent_loop.py:227-476`).
  - Built-ins: `single_turn_agent` (default) and `tool_agent`. Register with `@register(name)` (`:484-495`) or YAML; the dataset row selects the loop via `agent_name` (`:626-629`).
  - The docstring says the manager is meant to be replaceable by NeMo-Gym, SWE-agent, etc. (`:22-28`).
- **`ToolAgentLoop`** is a state machine PENDING → GENERATING ↔ PROCESSING_TOOLS → TERMINATED (`tool_agent_loop.py:48-52`, `:169-180`). It stops on budget, `max_assistant_turns`, `max_user_turns` or no tool call (`:289-308`).
  - **Tools execute inside the agent-loop worker process.** Upstream has no env-worker pool. Tool errors become tool text (`:455-492`).
  - Stateful `BaseTool.create/release` wrap **each call**, not the trajectory (`:485-496`).
- **Token-in-token-out ("Continuous Token")** is the only tokenization path (`agent_loop.py:261-271`; `V/verl/utils/tokenizer/continuous_token.py`).
  - Assistant tokens are appended verbatim (`:197-204`).
  - Tool, user and system turns are rendered as a **suffix diff** against a synthetic prefix (`:243-277`); messages must be append-only (`:327-341`).
  - Per-family junction fix-ups (e.g. the Qwen `\n` after `<|im_end|>`) are recorded as `inserted_token_ids` (`:540-578`).
  - `align_response_metadata` gives inserted and context tokens mask 0 / logprob 0.0 and checks lengths (`:365-419`).
  - Families: Qwen, GLM, MiniMax, Gemma4, GPT-OSS, DeepSeek, plus VL mixins.
- **Output format:** `AgentLoopOutput` holds `prompt_ids, response_ids, response_mask (1 gen / 0 tool), response_logprobs, routed_experts, reward_score, num_turns, extra_fields` (`agent_loop.py:88-115`).
  - The whole multi-turn trajectory is one sequence (`tool_agent_loop.py:183-213`).
  - v1 sets `loss_mask = response_mask`, unpadded (`agent_loop_tq.py:186-206`).
  - Overlong prompts are **left-truncated** (`agent_loop.py:454-463`).
  - Tool responses are cut by **characters** (`max_tool_response_length: 256`, middle, RY:217-220).
- **Multi-output trajectories:** v1 GRPO computes the advantage from the session's final output and **broadcasts** it to the other outputs (`V/verl/trainer/ppo/v1/utils.py:148-188`). This is the hook for Crucible compaction segments.
- **Failures and timeouts:**
  - **No timeouts anywhere in the agent loops** (grep `timeout` in `agent_loop/*.py` is empty).
  - v1 gathers session exceptions; the group becomes `failure`. In async it is evicted and refilled 1:1. In sync it stays sampleable, with missing trajectories padded unless `sync_refill_failed_groups` (`agent_loop_tq.py:134-151`; `replay_buffer.py:109-113`, `:531-539`).
- **Reward:** streamed from the agent loop to a random `RewardLoopWorker` Ray actor (`reward.num_workers: 8`) (`agent_loop.py:984-1046`; `V/verl/experimental/reward_loop/reward_loop.py:324-341`).
  - An agent loop may set `reward_score` itself, which skips this step (`agent_loop.py:749`, `:989`).
  - The remote RM client makes 16 retries with backoff, with `ClientTimeout(total=None)` right under a comment saying a timeout is safer (`reward_loop.py:157-195`; `:162-163` ✓).
  - A colocated RM is computed after sampling.
- **DAPO dynamic sampling:** `filter_groups.enable: False`, `metric: reward`, `max_inflight_gen_batches: 1`. `max_num_gen_batches` is **ignored by v1** (`trainer_base.py:205-211`).
  - The zero-std filter drops a group (`replay_buffer.py:261-315`).
  - **Sync:** evict k, refill 2k speculatively, then drain and discard the surplus (`:469-501`). **Async:** evict k, refill k (`:576-584`).
  - Soft overlong punishment is in the DAPO reward manager (`V/verl/experimental/reward_loop/reward_manager/dapo.py:33-41`).
  - `rollout.over_sample_rate` is **dead config** (`V/verl/workers/config/rollout.py:178-179`; nothing reads it).

## V7. Train–inference mismatch

- **Rollout logprobs** are recorded by default (`calculate_log_probs: True`) and stored as `rollout_log_probs` (`agent_loop.py:125-127`).
- **Two modes** (`rollout_correction.yaml:19-20` ✓):
  - **decoupled** (default): recompute π_old, three policies;
  - **bypass**: π_old := π_rollout, skipping the forward (`trainer_base.py:1642-1648` ✓).
- **Menu** (`V/verl/trainer/ppo/rollout_corr_helper.py`):
  - `rollout_is: token|sequence` with `rollout_is_threshold: 2.0`. A float means a **TIS** clamp; a `"lo_hi"` string means **IcePop** two-sided zeroing (`:572-604` ✓). Weights are detached, can be batch-normalized to mean 1, with a log-ratio `SAFETY_BOUND = 20.0` (`:75`).
  - `rollout_rs` rejection sampling on `token_k1|k2|k3`, `seq_sum_k*`, `seq_mean_k*`, `seq_max_k2|k3`, chainable (`:77-90`, `:197-377`). `seq_mean_k1` with bounds is the MIS-PO-like geometric filter.
  - Presets: `decoupled_token_is`, `decoupled_seq_is`, `decoupled_token_icepop`, `decoupled_geo_rs`, `bypass_ppo_clip`, `bypass_pg_is`, … (`V/verl/trainer/config/algorithm.py:213-686`).
  - Policy losses: vanilla, dppo_tv/kl, gspo, sapo, gpg, clip_cov, kl_cov, geo_mean, dro, **cispo**, bypass_mode (`core_algos.py:1285-2420`).
- **Diagnostics:** `training/rollout_probs_diff_{max,mean,std}` and a Pearson correlation (`V/verl/utils/debug/metrics.py:63-113` ✓); ESS, OOB ratio, KL/PPL/χ² (`rollout_corr_helper.py:660-785`).
- **FP32 LM head:** none, except `lm_head_precision: null` in the AutoModel engine (`V/verl/trainer/config/engine/automodel.yaml:58`).
- **Routing replay:** `engine.router_replay.mode: disabled|R2|R3` (Megatron, VeOmni) (`V/verl/trainer/config/engine/megatron.yaml:119-135`).
  - R3 needs `rollout.enable_rollout_routing_replay` and vLLM ≥ 0.22.0 (`vllm_async_server.py:440-454`).
  - R2 is incompatible with bypass and with R3 (`V/verl/utils/config.py:74-85`).
  - On resume, routing rows are appended for new tokens only (`llm_server.py:307-315`).
- Trainer-side temperature matches the rollout temperature (`trainer_base.py:1651-1657`).

## V8. Load balancing / long tail

- **`_balance_batch`:** Karmarkar–Karp with equal partition sizes over `24576·L+L²` across DP ranks (`trainer_base.py:1608-1632`; `seqlen_balancing.py:27-253`). v1 calls it **unconditionally**, ignoring `trainer.balance_batch` (`trainer_base.py:582` ✓), and drops v0's PP-bubble reordering.
- **Tail handling:**
  - sync is bounded by the slowest sample (logged as `agent_loop/slowest/*`, `agent_loop.py:1299-1310`);
  - async modes hide the tail with partial rollout + warmup over-provisioning;
  - hybrid lending moves idle trainer GPUs to rollout.
- **Absent:** length-aware scheduling, per-dataset concurrency caps, holding early short returns (length-bias control), straggler timeouts. v0 sends a vLLM `priority` that has no effect under the default `fcfs` (`agent_loop.py:1258-1262`; RY:104).

## V9. Checkpointing and fault tolerance

- **Layout:** `global_step_N/{actor,critic,data.pt,transfer_queue}` + `latest_checkpointed_iteration.txt` (`trainer_base.py:948-1029`).
  - FSDP saves per-rank `model_world_size_{W}_rank_{R}.pt`, so resume needs the same world size (`V/verl/utils/checkpoint/fsdp_checkpoint_manager.py:364-410`).
  - Megatron uses dist-checkpointing with optional `async_save`.
- **Async-mode state:** TransferQueue is saved (`trainer_base.py:1004-1012`). On resume, pending and running prompts are **re-issued** with their partials cleared (`_reissue_inflight_prompts`, `:910-946`).
- **Resume modes:** `auto | resume_path | disable`. An unknown value causes a NameError (`:873-876`).
- **Tracker file write is not atomic** despite the comment (`:1021-1025` ✓).
- **ESI expiry save** (capacity-block end) exists in v0 only (`ray_trainer.py:1665-1683`).
- **Beyond checkpoints: nothing.** Grep for `max_restarts|max_task_retries|health_check|fault_toler` in `verl` returns nothing. A dead worker kills the run.

## V10. Config surface (verl)

**Topology**
- `trainer.{nnodes: 1, n_gpus_per_node: 8, use_v1: true}`
- `trainer.v1.{trainer_mode: sync, colocate_async.num_warmup_batches: 1, separate_async.{num_warmup_batches: 1, parameter_sync_step: 4, hybrid_rollout.{enable_switch: false, switch_threshold_ratio: 0.4, adaptive_switch_threshold: true, …}}, sampler.{max_off_policy_threshold: 8, max_off_policy_strategy: drop, sync_refill_failed_groups: false}}` (VY:964-1003 ✓)
- `actor_rollout_ref.hybrid_engine: true` (VY:496)
- `rollout.{nnodes: 0, n_gpus_per_node}` (standalone)
- `reward.reward_model.enable_resource_pool: false`
- `transfer_queue.enable` (forced true in v1)

**Offload**
- `actor.fsdp_config.{param_offload: false, optimizer_offload: false, offload_policy: false, reshard_after_forward: true, fsdp_size: -1, model_dtype: fp32, use_no_sync_for_gradient_accumulation: false}` (VY:31-46)
- `model.{enable_gradient_checkpointing: true, enable_activation_offload: false, use_remove_padding: true}`
- `rollout.{gpu_memory_utilization: 0.5, free_cache_engine: true, load_format: dummy}`

**Training**
- `data.train_batch_size: 1024`
- `rollout.n: 1`
- `actor.{ppo_mini_batch_size: 256, ppo_micro_batch_size_per_gpu: null, use_dynamic_bsz: false, ppo_max_token_len_per_gpu: 16384, ppo_epochs: 1, loss_agg_mode: token-mean}`
- `critic.ppo_max_token_len_per_gpu: 32768`
- `ulysses_sequence_parallel_size: 1`
- Megatron TP/PP/CP/EP as in V2

**Rollout**
- `rollout.{tensor_model_parallel_size: 2, data_parallel_size: 1, expert_parallel_size: 1, max_num_seqs: 256, max_num_batched_tokens: 8192, enable_prefix_caching: true, ray_actor_max_concurrency: 1024, calculate_log_probs: true, agent.num_workers: 8, multi_turn.max_tool_response_length: 256}`

**Async and filtering**
- `algorithm.filter_groups.{enable: false, metric: reward, max_inflight_gen_batches: 1}`

**Mismatch**
- `algorithm.rollout_correction.{rollout_is: null, rollout_is_threshold: 2.0, rollout_rs: null, bypass_mode: false, loss_type: ppo_clip, rollout_is_batch_normalize: false}` ✓
- `engine.router_replay.mode: disabled`
- `rollout.enable_rollout_routing_replay: false`

**Weights**
- `rollout.checkpoint_engine.{backend: naive, update_weights_bucket_megabytes: 2048}` ✓
- `rollout.quantization: null`

**Checkpoint**
- `trainer.{save_freq: -1, resume_mode: auto, max_actor_ckpt_to_keep: null}`
- `actor.checkpoint.{save_contents: [model, optimizer, extra], async_save: false}`

## V11. MiMo fork (`MV/`): what differs

MiMo is an older upstream snapshot that already has v1, the TQ ReplayBuffer and Continuous Token (`MV/verl/trainer/ppo/v1/` ✓ has the same three modes). Its additions:

1. **`recipes/general` GeneralAgentLoop** (`MV/recipes/general/agent_loop.py:429`). It wraps a black-box agent (mimoagent):
   - **Threading:** the agent runs in a dedicated thread pool, because the default executor would deadlock Continuous-Token rendering (`:113-123`).
   - **Environment placement:** **one Ray actor per environment** (`DatasetEnvActor`, `num_cpus=1`), which owns a k8s pod (`MV/recipes/general/env_actor.py:75-76`). The actor is killed in `finally` (`agent_loop.py:971-978`).
   - **Timeouts** (absent upstream): `env_setup_timeout: 600`, `trajectory_timeout: 1800`, `reward_timeout: 1800` (`:441-445`).
   - **Infra taxonomy:** `_INFRA_ERROR_CATEGORIES` (`:78-87`). Those trajectories score `INVALID_REWARD_VALUE = -999.0` (`:76`, `:930-940`).
   - **Token budget:** a per-turn `max_tokens = min(per_turn, remaining)`, with budget exhaustion ending the trajectory at a message boundary (`:232-243`).
2. **TokenTrace and `chat_delta`** (`MV/recipes/general/token_trace.py:72-174`, `chat_delta.py:46-108`). Same intent as upstream Continuous Token, without per-family junction handling. **Pitfall:** missing logprobs are padded with `0.0`, i.e. probability 1 (`token_trace.py:119-128` ✓).
3. **−999 handling in GRPO** (`MV/verl/trainer/ppo/core_algos.py:306-350`). Invalid rows are imputed with the valid-row group mean and their advantage set to 0. That shrinks the std, so valid advantages inflate, and invalid tokens stay in the loss denominator and the KL term.
4. **Infra-aware DAPO classification** (`MV/verl/trainer/ppo/v1/replay_buffer.py:256-268`, `:310-339`). A group is judged only on its non-infra trajectories; ≤1 valid trajectory counts as no signal. It also adds per-harness pass-rate statistics (`:352-421`).
5. **Staleness** is the same ReplayBuffer, but `max_off_policy_threshold` must be an int > 0; null is not allowed (`replay_buffer.py:166-168` ✓). Failure outputs stamp `min/max_global_steps` with the current step (`recipes/general/agent_loop.py:685-713`).
6. **Extra operators:** `signed_rebalance.py`, `arvo_penalties.py`, `length_penalty.py`, already catalogued in ARCHITECTURE M12 "Takes".

---

# Part S: slime (`S/`)

## S0. Which code you are reading

- **One driver, `S/train.py:47-100`.** `train_async.py` does not exist.
  - The mode is chosen by `--rollout-function-path` (default `slime.rollout.sglang_rollout.generate_rollout`, SA:361-363) and `--flush-cache-interval` (default 1, SA:350-358).
- **Megatron is the only trainer:** `--train-backend` choices `["megatron"]` (SA:1707 ✓). `S/slime/backends/` holds only `megatron_utils/` and `sglang_utils/`.
- **SGLang is the only rollout engine,** behind `sglang_router`.
- **Patched dependencies:** `S/docker/patch/latest/` holds 7 patch files, about 5.8k lines (SGLang: pull_weights, deterministic, top_p, release_hicache; Megatron: base, sglang-aligned).

## S1. Topology and placement

- **Placement groups:** bundles `{"GPU": 1, "CPU": 1}`, strategy `PACK` (`S/slime/ray/placement_group.py:45-51` ✓). Each bundle is probed for (IP, GPU id) and sorted so ranks map to stable GPUs (`:24-42`, `:72-91`).
  - **colocate:** one PG of `max(actor, rollout)` GPUs (`:117-118`).
  - **disaggregated (default):** two PGs, "so a restarted trainer [can] resize without moving the serving engines" (`:126-138`).
  - `--rollout-external`: actor PG only.
- **Critic** always shares the actor PG and GPU count (`:138`; SA:2088-2090 ✓).
- **Fractions:** train actor `num_gpus_per_actor=0.4` (`placement_group.py:172` ✓); SGLang engine actor `num_gpus=0.2` (`S/slime/backends/sglang_utils/engine_group.py:182` ✓). 0.4 + 0.4 + 0.2 = 1, so actor + critic + engine fit on one GPU.
- **Engine processes:** the engine actor launches an SGLang server **subprocess**, which spawns one scheduler per TP rank (`S/slime/backends/sglang_utils/sglang_engine.py:183-186`, `:553-572`).
  - Per GPU (colocated GRPO): 1 Megatron process + 1 SGLang scheduler, plus wrapper and front-end processes on the base GPU. PPO adds a critic process.
- **Ref model: no process.** It is a **pinned-CPU weight copy** (`TensorBackuper`) swapped into the live Megatron model for the ref forward (`S/slime/utils/tensor_backper.py:11-38`; `S/slime/backends/megatron_utils/actor.py:127-138`, `:292-296`, `:460-471`). Optional `--ref-update-interval`.
- **Reward model:** not a trainer role. It is served as an extra SGLang model via `--sglang-config` with `update_weights: false` (`S/slime/ray/rollout.py:283-298`).
- **`RolloutManager`:** a CPU Ray actor.
- **`ServingCluster`:** a **detached named** actor that owns PGs, routers, engines, queue and weight-update lock (`placement_group.py:315-337`; `S/slime/ray/serving.py:34-77`).
- **Offload** (colocate implies `offload_train` and `offload_rollout`, SA:2117-2133; PPO forces `offload_train` even when disaggregated, SA:2139-2140 ✓):
  - **Trainer:** `torch_memory_saver` via `LD_PRELOAD` (CUDA VMM), with CPU backup (`S/slime/ray/actor_group.py:78-99`). `sleep()` = clear memory → **destroy NCCL groups** → pause; `wake_up()` reverses it (`actor.py:180-220`). Groups are made reloadable by monkey-patching `dist.new_group` (`S/slime/utils/reloadable_process_group.py:33-157`). Reported residual: 15–18 GB → 3–5 GB (`S/docs/en/blogs/release_v0.1.0.md:98`).
  - **Rollout:** `release_memory_occupation`, after **flushing KV** (`sglang_engine.py:335-337`). Two-stage onload: weights first (`engine_group.py:399-408`), then the weight sync, then KV + CUDA graphs (`:410-417`).
  - **Colocated loop:** generate → offload rollout → train (wake → train → sleep) → checkpoint → onload weights → pause admission → update_weights → resume admission → onload KV (`S/train.py:47-100`).
  - `--release-train` **kills** the Megatron actors during rollout and recreates them from disk (SA:165-173).
  - Optimizer CPU offload uses Megatron flags (`S/docs/en/advanced/parallelism-memory.md:115`).

## S2. Training parallelism (Megatron)

- **Parallel dims:** TP, PP, VPP, CP (+ hierarchical), EP and ETP go straight to `mpu.initialize_model_parallel` (`S/slime/backends/megatron_utils/initialize.py:37-57`).
  - Megatron SP flag; **no Ulysses** (grep empty).
  - Distributed optimizer forced on (`S/slime/backends/megatron_utils/arguments.py:156-157`); `variable_seq_lengths=True`.
  - CP is zigzag by default; `--allgather-cp` is restricted to DSA models.
- **Batches:**
  - `--rollout-batch-size` (prompts, required) × `--n-samples-per-prompt` (1).
  - `--global-batch-size` (None). Its help text says "of sample", but **the scheduler, loss normaliser and LR increment count rollouts** (`S/slime/utils/dp_schedule.py:99-104` ✓; `S/slime/backends/megatron_utils/model.py:542-548`).
  - `--num-steps-per-rollout` derives GBS (SA:2149-2157).
  - `--micro-batch-size` (1) is ignored under dynamic batching.
  - Grad accumulation = Megatron `forward_backward_func` over the step's micro-batches, then one `optimizer.step()` (`model.py:516-712`).
  - `--reset-optimizer-states` and `--use-stateless-adam` exist (SA:1139-1155).
- **Pack first, distribute second** (`dp_schedule.py:8-209`; computed on the RolloutManager, `S/slime/data/batch_builder.py:426-511`):
  1. **Group by rollout id** into steps of GBS rollouts. **Trailing rollouts are dropped** (`dp_schedule.py:135-149`).
  2. **Pack micro-batches:**
     - dynamic: `first_fit_pack(lengths, max_tokens_per_gpu×cp)`, in input order, not decreasing (`S/slime/utils/seqlen_balancing.py:159-177`);
     - `--balance-by-flops`: KK on `coeff·L+L²`, which may exceed the cap;
     - static: fixed chunks.
  3. **Align counts:** split bins until the count is a multiple of `dp×vpp_group`, so every rank runs the same number of micro-batches.
  4. **Distribute:** KK across DP with `--balance-data`, else strided.
- **Packing:** each micro-batch is one THD sequence with `cu_seqlens`, padded to `TP × data_pad_size_multiplier (128)` (`S/slime/backends/megatron_utils/data.py:14-134`).
  - `rollout_mask_sums` gives one token-mean per rollout even across micro-batches (`batch_builder.py:287-302`).
- **Log-prob recompute** reuses the training micro-batch schedule with `forward_only=True` (`model.py:352-513`). It is **skipped** when exactly on-policy (`can_reuse_log_probs_in_loss`, `actor.py:487-497`).
  - `--log-probs-max-tokens-per-gpu` is **dead**. It is set at SA:2047-2048 and listed in `S/slime/ray/training_recovery.py:96`, and read nowhere else (my grep ✓). This resolves a disagreement between the two slime surveys.
- **Data movement:** every TP/PP/CP rank `ray.get`s its DP shard on CPU (`S/slime/utils/data.py:303-318`; "not sure if this will be performance bottleneck", `actor.py:223`).

## S3. Rollout parallelism (SGLang)

- **Engines:** `--rollout-num-gpus-per-engine` (1) = SGLang TP×PP; engines = `rollout_num_gpus // per_engine` (`S/slime/utils/http_utils.py:233-242`). Multi-node engines are supported.
  - `--sglang-dp-size>1` requires DP attention (`S/slime/backends/sglang_utils/arguments.py:183-184`).
  - Every SGLang `ServerArgs` field is exposed as `--sglang-*` (`:41-121`).
  - Multi-model groups (actor / ref / RM / PD) come from `--sglang-config`.
- **Router:** one `sglang_router` per model, whose own health checks are disabled (`S/slime/backends/sglang_utils/deployment.py:18-62`).
  - SGLang default **cache-aware** policy, with slime's `router_balance_abs_threshold=10`, `rel=1.2` and request timeout 14400 s (`sglang_utils/arguments.py:28-46`).
  - `consistent_hashing` sends `X-SMG-Routing-Key: session_id` (`S/slime/rollout/sglang_rollout.py:213-217`).
- **Dispatch is per sample.** One task per group, fanned out into one task per sample; each POSTs `/generate` with `input_ids` and `return_logprob=True` (`sglang_rollout.py:142-155`, `:182-229`, `:404-414`). Continuous batching and radix prefix caching happen inside SGLang.
- **Concurrency:** `asyncio.Semaphore(sglang_server_concurrency × engines)`, default **512** per engine (`sglang_utils/arguments.py:47`; `sglang_rollout.py:95-97`). httpx pools are sharded at ≤128.
- **Dead code:** `dp_rank_context` computes a least-loaded DP rank and never sends it (`sglang_rollout.py:117-133`, `:335`).

## S4. Synchronous vs asynchronous

| Mode | Selection | Batch formation | In-flight at boundary |
|---|---|---|---|
| **Sync on-policy** (default) | default rollout fn | submit `over_sampling_batch_size` groups until `rollout_batch_size` pass the filter (`sglang_rollout.py:513-563`) | server-wide abort; results **discarded** (`:571`) |
| **Sync + partial rollout** | `--partial-rollout` (False) + over-sampling | same | aborted groups with tokens go **back to the buffer** with `start_rollout_id` (`:456-468`, `:772-774`); resume = re-prefill + continue with `max_new_tokens -= response_length` (`:46-62`); optional `--mask-offpolicy-in-partial-rollout` masks the old prefix (`:296-297`) |
| **Fully async (manager-local)** | `--rollout-function-path ...fully_async_rollout.generate_rollout_fully_async` | a background thread keeps `concurrency//n` groups in flight across steps (`S/slime/rollout/fully_async_rollout.py:67`, `:85`, `:197-218`); each call drains `rollout_batch_size` groups, surplus stays queued (`:295-301`) | generation continues during training; aborted groups requeued (`:256-262` ✓) |
| **Fully async (distributed, straw)** | `--rollout-data-transport straw` | one generation actor per CPU node, `Queue(maxsize=1)` each, scheduler waits for `demand` (`S/slime/rollout/fully_async_distributed.py:117-204`, `:289-292`) | as above; durable queue priority ready > partial > fresh, oldest version first (`S/slime/data/queue_data_source.py:1175-1197`) |
| **PipelineRL-style in place** | `--flush-cache-interval N≠1` | as fully async | paused **in place**, continues on old-weight KV (`S/docs/en/get_started/usage.md:423-455`). Needs separate GPUs, no offload (SA:2248-2257) |
| **Streaming generate** | custom generate fn `generate_streaming` | SSE; request-level abort keeps partial tokens (`S/slime/rollout/sglang_streaming_rollout.py:1-21`, `:139-183`) | top-p / R3 data only arrive on the terminal chunk |

- **Who triggers generation:** the driver calls `RolloutManager.generate` once per step (`S/slime/ray/rollout.py:303-319`). In async modes a background worker keeps producing between calls.
- **Buffers:**
  - `RolloutDataSourceWithBuffer` is an **unbounded** Python list. It serves the buffer before fresh prompts and pops FIFO, or oldest-version-first with `--buffer-sort-by-staleness` (`S/slime/data/data_source.py:190-265`; SA:562-569).
  - The fully-async output `queue.Queue()` is **unbounded on purpose**. Back-pressure comes from stopping top-up at `qsize ≥ max_concurrent` (`fully_async_rollout.py:92-97`, `:194-202`).
- **At a weight update** (`S/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py:102-143`; same in the tensor and disk-delta updaters):
  1. `weight_version += 1`;
  2. `flush = should_flush_cache(...)`, true on the first publication and every N-th (`S/slime/utils/weight_sync.py:4-13` ✓);
  3. `pause_generation(mode="abort" if flush else "in_place")`, then `flush_cache`;
  4. send;
  5. `continue_generation`.

  With the default `flush_cache_interval=1`, **every update aborts and flushes**; async workers requeue the whole group, which later re-prefills. Admission is also paused around every sync (`train.py:89-94`, `drain=False`).
- **Version tracking:** `Sample.weight_versions: list[str]`, appended **once per SGLang request that finishes with ≥1 new token** (`S/slime/utils/types.py:133`, `:487-488` ✓).
  - Staleness = current − min(versions); unknown if non-numeric or newer (`S/slime/utils/staleness.py:10-17` ✓).
  - Metrics `staleness/{mean,max,unknown_count}` are computed **only when the rollout fn path contains `"fully_async"`** (`staleness.py:6-7` ✓).
- **Staleness bound: none.** The docs say: "Ordering does not automatically discard stale samples." (`S/docs/en/advanced/rollout-scheduling.md:76`) and that TIS/SC "do not justify an unlimited stale queue" (`S/docs/en/advanced/rl-systems.md:41`). The only implicit bound is queue depth.
- **DAPO / over-sampling:**
  - `--over-sampling-batch-size` (None = `rollout_batch_size`) and `--dynamic-sampling-filter-path`. Stock filter `check_reward_nonzero_std` (std > 1e-6) (`S/slime/rollout/filter_hub/dynamic_sampling_filters.py:9-23`).
  - **Sync:** a dropped group triggers more sampling; `keep_when_insufficient` keeps it if dropping would force another round.
  - **Async:** rejected groups are replaced from the warm queue (`fully_async_rollout.py:301-321`).

## S5. Weight synchronization

| Condition | Updater | Mechanism |
|---|---|---|
| `--update-weight-mode delta` | `UpdateWeightFromDiskDelta` | shared-FS xor/overwrite deltas, zstd-1, per-tensor xxh3-128 checksum, atomic HF dir; engine `/pull_weights` **before** pausing (patched SGLang); not with colocate |
| `--update-weight-transport disk` | `UpdateWeightFromDisk` | full HF checkpoint per sync; optional per-host pull to NVMe before pausing |
| `--colocate` | `UpdateWeightFromTensor` | CUDA IPC `FlattenedTensorBucket` handles from the **pinned-CPU actor backup**; Gloo gather to the engine's first rank |
| default | `UpdateWeightFromDistributed` | one NCCL group per PP stage (`slime-pp_{rank}`, `1+Σ engine GPUs`), broadcast from rank 0 |

Source: `S/slime/backends/megatron_utils/update_weight/__init__.py:10-56`.

- **Buckets:** `--update-weight-buffer-size` default `512 * 1024**2` (SA:571-579 ✓).
  - Non-expert pass: TP all-gather → `convert_to_hf` (fused-QKV split, padding removal, optional FP8/INT4).
  - Expert pass: TP gather, then EP all-gather (`update_weight_from_distributed.py:162-247`).
  - A global lock serialises the PP groups. It polls with `sleep(0.1)` and needs one Ray RPC per bucket for metadata (`:249-274`, `:336-364`).
- **Resharding is on the receiver.** The trainer sends full HF tensors and SGLang loaders slice them into their own TP/EP layout.
- **IPC specifics:**
  - IPC buffers are **never reused**, because SGLang returns before its device copy is synchronized (`update_weight_from_tensor.py:40-44`).
  - MoE experts go by **P2P only to the ranks that need them** (`:195-284`; `expert_routing.py:295-325`).
  - The update runs with the memory saver disabled, because VMM and cudaIPC are incompatible (`actor.py:675-678`).
- **Quantization on the fly:** `fp8` → blockwise; `compressed-tensors` → INT4 with engine restore hooks. **Unknown methods (e.g. mxfp4) silently pass BF16** (`S/slime/backends/megatron_utils/megatron_to_hf/processors/__init__.py:10-22`).
- **Overlap:**
  - NCCL and IPC are **not** overlapped; engines are paused for the transfer.
  - Disk modes overlap publish and pull with generation, pausing only for the reload (`actor_group.py:257-268`; `update_weight_from_disk_delta.py:193-199`).
- **Stated cost:** GLM-4.5 355B-A32B bf16 sync 48 s; FP8 quantize + update 100 s (`release_v0.1.0.md:108`). Cost model `t_W ≈ S_W/B_W + t_0` (`rl-systems.md:11-15`).
- **Debug:** `--check-weight-update-equal`.

## S6. Agentic / multi-turn

- **Plug points:**
  - `--custom-generate-function-path`: per-sample `generate(args, sample, params) -> Sample | list[Sample]`, dispatched under the semaphore. Set `abort_mode="request"` for request-level cancellation (SA:510-519).
  - `--custom-rm-path` and `--group-rm`.
  - `--rollout-sample-hook-path`.
  - **Fan-out:** one rollout → several samples sharing `rollout_id`, counted once in the loss and kept in one step (`S/slime/ray/rollout.py:545-551`; `dp_schedule.py:127-133`).
- **`slime/agent`, the black-box harness stack:**
  - Claude Code / Codex harnesses run in **E2B sandboxes**, off the GPU pool, with 6 transient-RPC retries (`S/slime/agent/sandbox.py:160-271`; `S/slime/agent/harness/common.py:57-122`).
  - An aiohttp **adapter** speaks the Anthropic/OpenAI wire format. Every turn it **re-renders the full message list** with the chat template (`S/slime/agent/adapters/common.py:58-73`, `:341-342`), POSTs `input_ids` with logprobs, and records the turn.
  - Sticky routing uses `X-SMG-Routing-Key: <sid>`. A per-session turn cap returns 429.
  - On disconnect it calls `/abort_request`.
- **`TrajectoryManager`** (`S/slime/agent/trajectory.py`) classifies each turn against the held tokens:
  - **CLEAN:** exact extension; append with mask 0 / 1 and rollout logprobs (`:199-211`).
  - **REALIGN:** short drift inside the last response; overwrite that span, mask 0, logprob 0 (`:216-224`).
  - **FORK:** start a new Sample; `fork_threshold_tokens=1024` (`:270-271`, `:456-477`).

  This is **retokenization-tolerant capture, not strict token-in-token-out**. Shared prefixes are trained once, and every emitted sample gets the full reward.
- **Strict token-in-token-out examples:** retool, search-r1 and strands keep a running token list and append tool tokens with `loss_mask 0` (`S/examples/retool/generate_with_retool.py:249-380`). `Sample.append_response_tokens` enforces that trainable tokens carry logprobs (`types.py:271-299`, `:542-553`).
  - tau-bench is **not** token-in-token-out: it re-encodes assistant text and keeps no logprobs (`S/examples/tau-bench/trainable_agents.py:278-284`).
- **Failures:**
  - `Sample.Status`: PENDING, COMPLETED, TRUNCATED, ABORTED, FAILED. SGLang `length` → TRUNCATED, `abort` → ABORTED (`types.py:148-156`, `:490-496` ✓).
  - Coding-agent infra failures → ABORTED + `remove_sample=True` + reward 0.0 (`S/examples/coding_agent_rl/generate.py:316-333`). `remove_sample` **only zeroes the loss mask** (`batch_builder.py:282-283` ✓).
  - Env concurrency: `SWE_BOOT_CONCURRENCY` 16; agent budget 1800 s; eval 600 s (`generate.py:71-92`).
  - HTTP to SGLang: 60 retries, **no timeout** (`http_utils.py:189-256`).
- **Reward:** computed inline per sample in the same asyncio task (`sglang_rollout.py:357-376`). `rm_hub` covers remote_rm, deepscaler, dapo, math, f1, gpqa, ifbench.
  - Remote RM: 10 retries, then **raise** (`S/slime/rollout/rm_hub/__init__.py:22-52`). There is no separate reward-worker pool.
  - Group normalization happens at conversion (`batch_builder.py:209-233` ✓).

## S7. Train–inference mismatch

- **Rollout logprobs** are always requested and stored as `rollout_log_probs`.
- **π_old:** **recomputed** by default (`actor.py:498-512`). `--use-rollout-logprobs` (False) gives the bypass form. Recompute is skipped when on-policy (S2).
- **TIS:** `--use-tis` (False), `tis_clip_low` 0, `tis_clip` 2.0; weight = clamp(exp(π_old − π_rollout)) (`S/slime/backends/megatron_utils/loss.py:900-920`; `S/slime/utils/ppo_utils.py:25-35`).
- **IcePop:** `icepop_function` via `--custom-tis-function-path` (`loss.py:923-944`).
- **MIS menu** (example): `S/examples/train_infer_mismatch_helper/mis.py:82-451`. TIS level token / sequence / geometric; modes truncate / clip / mask; RS mask; veto at 1e-4; batch normalization; PPL / KL / χ² metrics; CP-aware.
- **OPSM:** `--use-opsm`, δ 1e-4. Masks a sequence if A < 0 and mean(π_old − π_θ) > δ (`ppo_utils.py:226-264`).
- **Objectives:** grpo, gspo, cispo, reinforce_plus_plus(_baseline), ppo (SA:1047-1062).
- **Always logged:** `train_rollout_logprob_abs_diff` (`loss.py:1327-1345`). With TIS also `tis_clipfrac`, `ois`.
- **FP32 LM head: none.** The docs say "the LM head stays bf16 on both train and rollout — matching precision, not fp32, is what aligns" (`S/docs/en/advanced/reproducibility.md:62`).
  - Bitwise alignment for GLM-5: < 1e-6 via deterministic SGLang + Megatron and DeepGEMM batch-invariant FP8 (`reproducibility.md:54-78`; `S/slime/backends/megatron_utils/alignment/`).
- **Routing replay:**
  - **R2** `--use-routing-replay` (`S/slime/utils/routing_replay.py:173-219`).
  - **R3** `--use-rollout-routing-replay`. Routed experts come back from SGLang as int32 `[tokens-1, layers, topk]`; continuations append rows only and refuse missing history (`sglang_rollout.py:190-199`; `types.py:429-475`).
  - **Top-p candidate replay** (MiMo-style).
  - No routing-confidence gate.

## S8. Load balancing / long tail

- **Training:** `dp_schedule` (S2), with Karmarkar–Karp copied from verl (`seqlen_balancing.py:1-2` ✓).
- **Rollout:**
  - per-sample admission keeps continuous batching full;
  - cache-aware router (the docs warn it can unbalance; `--router-balance-abs-threshold 0` forces balance, `usage.md:148`);
  - over-sampling + tail abort (sync);
  - partial rollout keeps the tail's prefix;
  - fully async never waits for the slowest sample.
- **Absent:** length-aware scheduling, per-dataset caps, short-return holding, straggler timeouts.
- **Metrics:** `perf/longest_sample_tokens_per_sec`, `zero_std/count_*`, prefix-cache hit rate (`S/slime/observability/rollout_metrics.py:61-214`).

## S9. Checkpointing and fault tolerance

- **Megatron checkpoints:** `torch_dist` + `--async-save`. Rollout state is saved too: data cursor + partial buffer (`global_dataset_state_dict_{id}.pt`, `data_source.py:136-182`) and the batch builder. Straw writes an atomic commit file (`S/slime/data/checkpoint.py:62-109`).
- **Trainer fault tolerance** (`S/slime/ray/training_recovery.py:29-140`; `serving.py:34-180`; `S/docs/en/advanced/fault-tolerance.md`):
  - the detached `ServingCluster` survives driver death;
  - a resubmitted trainer may change TP/PP/CP/EP/VPP / micro-batch / offload;
  - it loads the last checkpoint and **replays retained batches** re-partitioned for the new DP layout, without regenerating (`rollout.py:321-376`);
  - producers stay paused until restored weights are pushed.
- **Engine health** (`S/slime/utils/health_monitor.py:10-154`):
  - concurrent `/health_generate` probes; failed or wedged engines are retired;
  - restart **just before the next weight update** (`actor.py:632-637` → `serving.py:256-266`);
  - interval / timeout / first-wait default 600 s (SA:666-683).
- **Not covered:** loss of the ServingCluster or the Ray cluster. KV is never restored.

## S10. Config surface (slime)

| Key | Default | Cite |
|---|---|---|
| `--actor-num-nodes` / `--actor-num-gpus-per-node` | 1 / 8 | SA:40-43 |
| `--num-gpus-per-node` | 8 | SA:62-70 |
| `--rollout-num-gpus` / `--rollout-num-gpus-per-engine` | None (= actor under colocate) / 1 | SA:45-61 |
| `--colocate` / `--offload` / `--offload-train` / `--offload-rollout` | False / False / derived / derived | SA:71-101, 2115-2144 |
| `--update-weight-mode` / `--update-weight-transport` | `full` / `nccl` | SA:144-164 |
| `--update-weight-buffer-size` | `512 * 1024**2` | SA:571-579 ✓ |
| `--flush-cache-interval` | 1 | SA:350-358 |
| `--rollout-function-path` | `slime.rollout.sglang_rollout.generate_rollout` | SA:361-363 |
| `--rollout-batch-size` / `--n-samples-per-prompt` | required / 1 | SA:771-781 |
| `--global-batch-size` / `--num-steps-per-rollout` / `--micro-batch-size` | None / None / 1 | SA:787-798 |
| `--use-dynamic-batch-size` / `--max-tokens-per-gpu` | False / None | SA:827-847 |
| `--balance-data` / `--balance-by-flops` | False / False | SA:799-825 |
| `--over-sampling-batch-size` / `--dynamic-sampling-filter-path` | None / None | SA:464-489 |
| `--partial-rollout` / `--mask-offpolicy-in-partial-rollout` | False / False | SA:491-508 |
| `--buffer-sort-by-staleness` | False | SA:562-569 |
| `--rollout-data-transport` | `object-store` (`straw`, `nixl`) | SA:590-600 |
| `--sglang-server-concurrency` | 512 | `sglang_utils/arguments.py:47` |
| `--router-balance-abs/rel-threshold` | 10 / 1.2 | `sglang_utils/arguments.py:46` |
| `--use-rollout-logprobs` / `--use-tis` / `--tis-clip` / `--tis-clip-low` | False / False / 2.0 / 0 | SA:1157-1192 |
| `--use-opsm` / `--opsm-delta` | False / 1e-4 | SA:1219-1229 |
| `--use-routing-replay` / `--use-rollout-routing-replay` | False / False | SA:1201-1211 |
| `--advantage-estimator` / `--eps-clip` / `--eps-clip-high` | grpo / 0.2 / None | SA:1005-1062 |
| `--rollout-health-check-interval/-timeout/-first-wait` | 600 / 600 / 600 | SA:666-683 |
| `--save` / `--save-interval` / `--async-save` | None / None / store_true | SA:959-962 |

---

# Part C: comparison and recommendations

## (a) verl vs slime on every dimension

| Dimension | verl (`V/`, v1 trainer) | slime (`S/`) |
|---|---|---|
| **1. Topology** | Ray STRICT_PACK PG per node; all roles in a pool fused into **one process per GPU**; HYBRID / COLOCATED / STANDALONE rollout; vLLM runs as a separate mp process on the trainer's GPUs | Ray PACK bundles; colocate = 1 PG, disaggregated = 2 PGs (default); actor 0.4 + critic 0.4 + engine 0.2 GPU; detached ServingCluster |
| Offload | engine contexts auto-offload params / optimizer / grads; vLLM sleep level 2 (1 for LoRA / MTP / NPU); weights-then-KV resume order with a barrier | torch_memory_saver VMM + NCCL group destroy / rebuild; SGLang release with KV flush; two-stage onload; `--release-train` kills trainers |
| Ref / RM | ref fused (LoRA: adapter off); RM in its own pool, the global pool, or COLOCATED | ref = pinned-CPU weight swap (no process); RM = extra SGLang model, frozen |
| **2. Training** | FSDP / FSDP2 (+Ulysses), Megatron (TP/PP/VPP/CP/EP/ETP, dynamic CP), VeOmni, TorchTitan, AutoModel | Megatron only (TP/PP/VPP/CP/EP/ETP, SP); no FSDP, no Ulysses |
| Batch units | `ppo_mini_batch_size` in **prompts** (×n inside) | `global_batch_size` help says samples; scheduler counts **rollouts** |
| Dynamic batching | token budget `16384`/GPU; KK on `24576L+L²`; MAX-reduced micro count; PP-bubble interleave | first-fit (not decreasing) on `max_tokens_per_gpu×cp`; optional KK by FLOPs; counts aligned to DP×VPP |
| Loss normalization | global-token across DP per mini-batch (`agg_loss`) | per-rollout `rollout_mask_sums` |
| π_old recompute | once per batch; skipped in bypass | same schedule as training; skipped when exactly on-policy |
| **3. Rollout** | vLLM / SGLang / TRT-LLM; TP2 default; one shared Ray-actor router (sticky + least-in-flight); per-request dispatch; no agent-level semaphore; `max_num_seqs` 256 | SGLang only; one sglang_router per model (cache-aware); per-sample dispatch; semaphore 512×engines |
| **4. Sync / async** | `sync`, `colocate_async` (partial rollout on shared GPUs), `separate_async` (+ GPU lending), legacy one-step-off / fully-async | sync, sync+partial, fully async (local / distributed), in-place pause (separate GPUs only), streaming |
| Staleness bound | **yes:** `max_off_policy_threshold: 8`, `drop`/`wait`, measured from prompt submission | **none**; metrics only (fully-async mode), oldest-first ordering |
| Version granularity | min/max `global_steps` per trajectory | one version per finished SGLang request; none for adapter agents |
| In-flight at update | abort → keep tokens → re-prefill (prefix cache cleared); admission gate parks late requests | default abort + flush → requeue group → re-prefill; or in-place pause with stale KV |
| **5. Weight sync** | CheckpointEngine: naive (IPC), nccl, nixl, mooncake, kimi, delta; 2048 MB buckets; full materialization; FP8 at load; generation stops during disaggregated sync | IPC from pinned-CPU backup, NCCL per PP stage, disk, disk-delta; 512 MiB buckets; receiver reshards; P2P expert plan; FP8 / INT4 on the fly |
| **6. Agentic** | AgentLoop per trajectory; tools in-process; Continuous Token (strict token-in-token-out, family fix-ups); multi-output per session; **no timeouts**; streaming reward workers | custom generate fn; E2B harness adapters with CLEAN / REALIGN / FORK drift handling; strict token-in-token-out in examples; inline reward; ABORTED / `remove_sample` |
| DAPO in async | evict k, refill k (stale / filter / failed) | rejected groups replaced from warm queue |
| **7. Mismatch** | TIS / IcePop / k1–k3 RS (incl. geometric), decoupled vs bypass, CISPO / GSPO losses; `rollout_probs_diff`; R2 / R3; no FP32 head (AutoModel only) | TIS, IcePop fn, MIS example (token / seq / geo, veto), OPSM, SC; `train_rollout_logprob_abs_diff`; R2 / R3 + top-p replay; bitwise-aligned GLM-5 path; no FP32 head (by design) |
| Critic GAE | **skips observation tokens** | no mask in batched GAE |
| **8. Long tail** | KK DP balance (unconditional in v1); partial rollout; warmup batches; GPU lending; no length-bias control | KK / first-fit; over-sample + abort; partial rollout; fully async; no length-bias control |
| **9. Fault tolerance** | checkpoint + TQ state + in-flight re-issue; nothing else (no restarts, no health checks) | detached serving, replay-on-restart with resharding, engine health monitor + restart before sync |
| **10. Config** | Hydra yaml + dataclasses (some defaults drift) | argparse (`arguments.py`) + `--sglang-*` passthrough |
| Patches / deps | stock vLLM/SGLang; TransferQueue 0.1.11 | ~5.8k lines of SGLang/Megatron patches |
| Single-GPU fit (lab 5090) | yes: FSDP2 + vLLM TP=1, `colocate_async` | colocate sync only; async-with-colocate refused; 8-GPU defaults |

## (b) Recommendations for Crucible M12

The decision follows from DESIGN §10.3:
- one shared lab GPU;
- colocated async with partial rollout;
- a per-span version cap τ (default 4);
- token-in-token-out records;
- a selectable mismatch menu;
- a critic mode for group size 1;
- compaction segments.

Only verl v1 runs that topology today.

### DEPEND

| What | How | Why / caveats |
|---|---|---|
| **verl v1 trainer, `trainer.v1.trainer_mode=colocate_async`** (`V/verl/trainer/ppo/v1/trainer_colocate_async.py:19-50`) | pin 5eb4826; FSDP2 engine; vLLM `tensor_model_parallel_size=1`, `gpu_memory_utilization` tuned for 32 GB; `rollout.n=8`; `sampler.max_off_policy_threshold=4` (DESIGN τ); `rollout_correction` preset selected from Crucible's `mismatch.py` choice | The one topology match. Avoid v0 (deprecated, `main_ppo.py:183-191`). Accept the TransferQueue 0.1.11 dependency. |
| **verl agent-loop plug-in point** (`V/verl/experimental/agent_loop/agent_loop.py:484-495` register; `AgentLoopOutput` `:88-115`) | a `CrucibleAgentLoop` drives a World (environments stay on CPU, off the GPU pool), calls the policy through verl's `LLMServerClient`, and returns `prompt_ids / response_ids / response_mask / response_logprobs / routed_experts` plus `reward_score` (skips verl's reward loop, `:749`, `:989`). The full `crucible.rollout/v1` record goes in `extra_fields`. | Token-in-token-out is native. One output **per compaction segment**, using the v1 multi-output broadcast (`V/verl/trainer/ppo/v1/utils.py:148-188`). |
| **verl advantage registry** (`V/verl/trainer/ppo/core_algos.py:116` `register_adv_est`) | register `crucible_turn` (M6 turn advantages → call-id spans), `crucible_signed_span`, and a `gae` variant for critic mode | Credit enters without forking the trainer. The built-in GAE already skips observation tokens (`core_algos.py:254-256`). |
| **verl rollout-correction + policy losses** (`rollout_corr_helper.py`, `core_algos.py:1285-2420`) | configure, not re-implement: TIS (float threshold), IcePop (`"lo_hi"`), MIS-PO via `rollout_rs: seq_mean_k1` + token bounds, CISPO / GSPO losses | Covers 3 of the 5 `mismatch.py` options plus CISPO / GSPO / Dr.GRPO-style `loss_agg_mode`. |

### PORT (Apache-2.0; keep headers and the verl Notice line)

| What | File:line | Into | Note |
|---|---|---|---|
| Token-budget micro-batching + KK balancing | `V/verl/utils/seqlen_balancing.py:27-505` | `crucible/train/batching.py` (bridge-agnostic, CPU-tested) | used by prime-rl/TRL bridges too |
| Rollout-grouped DP schedule (all samples of one rollout in one step; equal micro counts) | `S/slime/utils/dp_schedule.py:51-209` (+ `S/tests/test_dp_schedule.py`) | same | correct unit for compaction segments; **fix** silent tail drop (`:135-149`) and use first-fit-decreasing |
| Flush-vs-in-place policy as a pure function | `S/slime/utils/weight_sync.py:4-13` | `topology.py` | 13 lines; makes the in-flight policy explicit |
| Staleness age = current − min(version), unknown handling | `S/slime/utils/staleness.py:10-29` | `staleness.py` metrics | apply to Crucible's **per-span** versions; add the τ drop rule slime lacks |
| Mismatch corrections + metrics not in verl | `S/examples/train_infer_mismatch_helper/mis.py:82-451`; `S/slime/utils/ppo_utils.py:226-264` (OPSM) | `mismatch.py` | veto, geometric-level clipping, χ² metrics, OPSM. **Write KPop and DIS ourselves**; neither framework has them. |
| `rollout_probs_diff` + Pearson | `V/verl/utils/debug/metrics.py:63-113` | `monitors.py` | DESIGN §10.3 point 4 "always logged" |
| Sticky + least-in-flight router | `V/verl/workers/rollout/router.py:133-298` | only if Crucible runs its own engine pool outside verl | replace `hash()` with `crc32` |
| Infra-aware DAPO group classification | `MV/verl/trainer/ppo/v1/replay_buffer.py:256-268`, `:310-339` | `objectives.py` dynamic sampling | judge groups on valid rollouts only; Crucible additionally keeps all-fail groups that carry κ(1−O)Φ |

### PATTERN

- **Partial-rollout resume client** (`V/verl/workers/rollout/llm_server.py:225-361`):
  - **Gap:** it keeps only `min/max_global_steps`.
  - **Change:** at each `aborted` boundary, close the current span and open a new one with the server's `global_steps`. This yields `Turn.policy_version` per span. A single model turn that spans an update becomes two spans of one `call_id`, so `Turn` needs either a `version_spans: list[(start_token, version)]` or a split. Decide this in M12.
- **Admission gate around weight updates** (`vllm_async_server.py:1048-1187`) and the colocated memory order weights → stream → offload → barrier → KV (`V/verl/workers/engine_workers.py:776-837`). The slime equivalent is the two-stage onload (`S/slime/backends/sglang_utils/engine_group.py:399-417`).
- **ReplayBuffer eviction matrix** (`V/verl/trainer/ppo/v1/replay_buffer.py:64-613`): drop vs wait, failed-group refill, oldest-first. Change the age rule from *submission step* to *oldest generated span*, which is DESIGN's GLM w₀ rule, so a prompt queued but not started does not age.
- **TrajectoryManager CLEAN / REALIGN / FORK** (`S/slime/agent/trajectory.py:116-261`) for opaque harnesses behind the M10 capture proxy:
  - Crucible currently masks the whole rollout on `prefix_break`.
  - Offer REALIGN (mask only the drifted span) and FORK (new segment) behind a flag.
  - Never pad logprobs; see the MiMo pitfall.
- **No-op padding samples** for divisibility after masking (`V/verl/trainer/ppo/padding_utils.py:151-175`).
- **Fault tolerance:**
  - slime's detached serving owner + replay-on-restart (`S/slime/ray/serving.py:34-180`, `S/slime/ray/rollout.py:321-376`);
  - the health monitor with restart-before-sync (`S/slime/utils/health_monitor.py:10-154`);
  - verl's TQ checkpoint + in-flight re-issue (`trainer_base.py:899-946`). This is the trainer-side twin of Crucible's World checkpoint/resume (DESIGN §10.4 point 5): a resumed prompt should resume its **World checkpoint**, not restart the episode.
- **Ref as a pinned-CPU weight swap** (`S/slime/utils/tensor_backper.py:11-38`) if KL-to-ref is ever needed on one GPU.
- **MiMo env-per-actor, timeouts and infra taxonomy** (`MV/recipes/general/env_actor.py:75-76`; `agent_loop.py:76-87`, `:441-445`). Crucible's runtime already covers the env side; take the three timeouts, because upstream agent loops have none.
- **Delta weight sync** (verl `delta_checkpoint_engine.py`, slime disk-delta) for the later disaggregated phase.

### Things neither has, which M12 must build

1. **A per-span policy version**, and staleness measured from the oldest generated span.
2. **Length-bias control:** per-dataset concurrency caps and holding early short returns. verl has no agent-level semaphore at all.
3. **Windowed FIFO** (MiniMax W = 0.3N).
4. **KPop** binary-KL masking and **DIS** (π_rollout denominator with two-sided zeroing). DIS can be approximated with verl bypass mode, but see pitfall P6.
5. **An FP32 LM-head switch** for FSDP + vLLM. slime argues for matched bf16 instead (`S/docs/en/advanced/reproducibility.md:62`); measure `rollout_probs_diff` before and after.
6. **A routing-confidence gate** for R3.
7. **Cross-segment GAE** for compaction (DESIGN §10.4).
8. **Per-call timeouts** in the agent loop, plus an infra mask that removes failed rollouts from the **group statistics**, not just the loss (see P3).
9. **Truncation as mask or Φ-bootstrap.** verl's DAPO overlong handling is reward-side soft punishment only (`V/verl/experimental/reward_loop/reward_manager/dapo.py:33-41`).

### Exit-gate mapping (ARCHITECTURE M12)

**Lab-5090 gate:** one colocated async GRPO run on the toy pack with staleness and `rollout_probs_diff` monitors. It runs on verl `colocate_async` with:
- the Crucible agent loop;
- the Crucible advantage estimator;
- the threshold set to τ;
- `rollout_correction` set from `mismatch.py`.

**Contract tests:**
- **Token-in-token-out round trip:** compare `AgentLoopOutput` built from a `crucible.rollout/v1` record against the original ids and logprobs.
- **Staleness drop rule:** unit-test the ported age function on synthetic span versions.
- **Mismatch corrections:** call the ported functions on synthetic ratio distributions.

None of these needs a GPU.

## (c) Pitfalls and bugs found

**verl (upstream)**

| # | Pitfall | Where |
|---|---|---|
| P1 | Defaults drift between yaml and dataclass: `use_dynamic_bsz` (false vs True) and `use_no_sync_for_gradient_accumulation` (false vs True). Behaviour differs between Hydra and programmatic configs. | VY:83 vs `V/verl/workers/config/engine.py:100` ✓; VY:46 vs `engine.py:288` |
| P2 | Version granularity is min/max per trajectory, and age counts from **submission**. Queued-not-started prompts age; mixed-version responses are invisible. | `llm_server.py:333-359` ✓; `replay_buffer.py:520-529` ✓ |
| P3 | No timeouts in agent loops. Reward HTTP has `ClientTimeout(total=None)` under a comment saying a timeout is safer. | grep empty; `reward_loop.py:162-163` ✓ |
| P4 | "Deterministic" routing uses salted `hash()`. | `router.py:197` ✓ |
| P5 | Every resume re-prefills the whole context (prefix cache cleared at abort). For 100k+ token agent trajectories, each abort costs a full prefill, which argues for small τ and few aborts. | `vllm_async_server.py:1135-1138` |
| P6 | v1 bypass mode only swaps `old_log_probs`. Unlike v0, it does not switch the actor to the `bypass_mode` loss, so the `rollout_rs`/`rollout_is` settings may be silently ignored. Inferred from the code; **test before relying on it.** | `trainer_base.py:1642-1648` ✓ vs `ray_trainer.py:1576-1583` |
| P7 | v1 ignores `trainer.balance_batch` and dropped v0's PP-bubble reorder. | `trainer_base.py:582` ✓ |
| P8 | Dead config: `over_sample_rate`, `multi_stage_wake_up`, and in v1 `filter_groups.max_num_gen_batches` and `checkpoint_manager_class`. | `V/verl/workers/config/rollout.py:178-179`, `:219`; `trainer_base.py:205-211`, `:383` |
| P9 | Stateful tools are created and released per call; tool truncation counts characters; overlong prompts are left-truncated, dropping the system prompt. | `tool_agent_loop.py:485-506`; `agent_loop.py:454-463` |
| P10 | The checkpoint tracker write is not atomic; an unknown `resume_mode` causes a NameError; the ESI expiry save exists only in v0; there is no fault tolerance beyond restart. | `trainer_base.py:1021-1025` ✓, `:873-876` |
| P11 | One fused process per GPU, so one OOM kills actor, ref and critic together. | `single_controller/ray/base.py:1008-1048` |
| P12 | `ppo_mini_batch_size` is in prompts, ×n internally: a classic 8× error when porting configs. | `trainer_base.py:1829-1830` |
| P13 | Legacy fully-async `MessageQueue` drops the oldest sample when full. | `message_queue.py:66-83` |
| P14 | Mismatch correction is **off by default** (`rollout_is: null`), so a run without an explicit preset trains uncorrected. | `rollout_correction.yaml:6` ✓ |

**MiMo fork**

| # | Pitfall | Where |
|---|---|---|
| P15 | TokenTrace pads missing logprobs with `0.0` (probability 1), corrupting every IS ratio on those tokens. Crucible's "mask, never guess" rule forbids porting this. | `MV/recipes/general/token_trace.py:119-128` ✓ |
| P16 | −999 imputation with the group mean shrinks the std and inflates valid advantages. Invalid tokens stay in the loss denominator and the KL term. | `MV/verl/trainer/ppo/core_algos.py:306-350` |
| P17 | `trajectory_timeout` cannot cancel the agent thread; it relies on `model.stopped` and killing the env actor. | `MV/recipes/general/agent_loop.py:896-913` |

**slime**

| # | Pitfall | Where |
|---|---|---|
| P18 | **No staleness cap.** Requeued ABORTED or partial groups can be trained at any lag. | `rollout-scheduling.md:76`; `fully_async_rollout.py:256-262` ✓ |
| P19 | Versions are recorded per finished request: lost for in-place pauses spanning an update, for streaming aborts, and for all adapter agent samples. Staleness metrics exist only in fully-async mode. | `types.py:487-488` ✓; `staleness.py:6-7` ✓; `trajectory.py:28-38`, `:248-261` |
| P20 | Infra failures (`remove_sample`) still enter the GRPO **group mean and std as reward 0**; only the loss mask is zeroed. This violates "masked, never 0" for the siblings' advantages. | `batch_builder.py:209-233`, `:282-283` ✓ |
| P21 | When sample counts per group differ (fan-out), group normalization falls back to **one batch-wide group** (`view(-1, N)`). | `batch_builder.py:217-222` ✓ |
| P22 | Deterministic infra failures marked ABORTED are retried forever in async. Remote-RM failure (raise after 10 retries) kills the async worker. Rule RMs run on the event loop. | `S/examples/coding_agent_rl/generate.py:188-192`; `rm_hub/__init__.py:46-49`; `fully_async_rollout.py:247-249`, `:296-297` |
| P23 | An abort is rendered to black-box harnesses as Anthropic `end_turn`, so a truncated answer looks final and is not marked ABORTED. | `S/slime/agent/adapters/anthropic.py:173-178` |
| P24 | The manager-local async ABORTED check iterates only the top level of the result, so nested fan-out groups are missed; the distributed path uses `iter_samples`. Static reading, not executed. | `fully_async_rollout.py:257` ✓ vs `fully_async_distributed.py:362-365` |
| P25 | `global_batch_size` counts samples in the help text and rollouts in the scheduler; trailing rollouts are silently dropped; `first_fit_pack` is not decreasing; `--balance-by-flops` can exceed the token cap and OOM. | SA:784-787; `dp_schedule.py:99-104` ✓, `:135-149`; `seqlen_balancing.py:159-177` |
| P26 | Dead knobs and code: `--log-probs-max-tokens-per-gpu` (my grep ✓) and `dp_rank_context`. | SA:2047-2048; `sglang_rollout.py:117-133`, `:335` |
| P27 | Unknown quantization (mxfp4) silently sends BF16. | `megatron_to_hf/processors/__init__.py:21-22` |
| P28 | PPO forces trainer offload even when disaggregated, paying a VMM pause and an NCCL rebuild every step. | SA:2139-2140 ✓ |
| P29 | HTTP to SGLang has 60 retries and no timeout. Health checks default to 600 s first-wait, so in sync mode detection relies on the end-of-rollout forced check. | `http_utils.py:189-256`; `health_monitor.py:88-113` |
| P30 | Fully async + `--colocate` with the default flush is not rejected by validation, although `train.py` offloads engines right after `generate` while the async worker keeps running. Static reading, not executed. | SA:2248-2252; `train.py:53-54`; `rollout.py:437-444` |
| P31 | The `num-gpus-per-node` default of 8 feeds the NUMA rank math; single-GPU boxes must pass `--num-gpus-per-node 1`. | SA:62-70; `S/slime/ray/train_actor.py:83` |
| P32 | About 5.8k lines of SGLang/Megatron patches: `/pull_weights` and the deterministic paths exist only in the patches. A version-pinning burden for any DEPEND. | `S/docker/patch/latest/` |

**Shared lessons for Crucible**
- **Mismatch correction:** both frameworks ship it off by default (P14; `--use-tis` False). Crucible's config must choose one explicitly and log `rollout_probs_diff` regardless.
- **Masking infra failures:** both get it subtly wrong (P16, P20). The Crucible bridge must remove a masked rollout from the **group statistics** and the **loss denominator**, not only zero its advantage or loss mask. Then it renormalizes over valid siblings, and drops the group if fewer than 2 are valid.
- **Per-span versions:** neither tracks them (P2, P19). M12's resume client must emit spans, or `oldest_version()` in `rollout.py` will under-report staleness for every rollout that crossed an update.
