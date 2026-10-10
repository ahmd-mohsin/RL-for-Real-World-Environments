# M12 research 04: weight sync (checkpoint-engine), OpenRLHF, TRL, rLLM

- **Date:** 2026-10-09
- **Scope:** Crucible M12 (`train`), the RL-infrastructure layer (DESIGN §10.3–10.4, ARCHITECTURE "M12 `train`").
- **Context read first:** DESIGN §10.3 ("RL infrastructure, trainer side", points 1–8) and §10.4; ARCHITECTURE M12 (`rollout.py`, `topology.py`, `staleness.py`, `mismatch.py`, `objectives.py`, exit gate); `docs/research/2026-10-09-agentic-rl-infra-survey.md`; `src/crucible/train/rollout.py`.
- **Method:** I read the code at the pinned commits (`00-source-pins.txt`). Every claim cites `file:line`, relative to that repo's root. README or doc claims are labelled **[README]** or **[doc]**. Where something does not exist, the search used is named. Inferences not verified in code are labelled **(inference)**.

| Repo | Pin | License | Version |
|---|---|---|---|
| `infra-moonshotai-checkpoint-engine` | 84631a2 (2026-09-15) | MIT (`LICENCE`, British spelling) | — |
| `infra-openrlhf-openrlhf` | dc2a7ad (2026-09-17) | Apache-2.0 | `version.txt` = 0.11.3; `vllm==0.29.0` (setup.py:77) |
| `infra-huggingface-trl` | 72cf37d (2026-10-09) | Apache-2.0 | `VERSION` = 1.16.0.dev0; extra `vllm>=0.21.0,<=0.31.0` (pyproject.toml:84) |
| `infra-rllm-org-rllm` | 4e828c8 (2026-10-08) | Apache-2.0 | see §D |

**Bottom line:**
1. **Exit-gate backend: TRL `GRPOTrainer` with `vllm_mode="colocate"`.** It is the only one of the four that runs as-is on one GPU with a multi-turn external environment, with token-in-token-out tool loops, rollout logprobs, mismatch-IS correction and DAPO/Dr.GRPO/CISPO/GSPO-style losses. It is **synchronous**, though. Crucible's "colocated async" requirement has to come from a thin Crucible layer on top: a `rollout_func` that drives Worlds concurrently, with version stamps and a staleness gate (§G).
2. **TRL `AsyncGRPOTrainer` and OpenRLHF async both need ≥ 2 GPUs.** TRL needs a separate vLLM server reached over NCCL; OpenRLHF async gives vLLM its own placement-group bundle. Both are the patterns to copy for M12's async control flow, not the exit-gate runtime.
3. **checkpoint-engine is a multi-GPU and multi-node weight-plane component.** On one GPU it adds nothing over in-process `load_weights` or CUDA IPC. Keep it as a PATTERN (bucketed double-buffered broadcast, ZMQ state machine into a vLLM worker extension) and a later DEPEND for disaggregated M12.

---

## A. checkpoint-engine (Moonshot): the weight-update protocol

### A.1 What it is
- **[README]:** "a simple middleware to update model weights in LLM inference engines", in place. Kimi-K2 (1T parameters) across thousands of GPUs takes "about 20s" (README.md:2-3).
- **Core:** `ParameterServer` (checkpoint_engine/ps.py:176), "a service colocated with inference engines" [README:14]. It runs **one process per inference GPU** (`_local_rank = rank % gpu_count`, ps.py:202; `set_device(local_rank)`, ps.py:231-232).
- **Its own control plane:** a `torch.distributed.TCPStore` on `MASTER_PORT+1` (ps.py:166-173, 268-274), separate from the inference engine's.

### A.2 Data model: who owns what
- **Registration:** a checkpoint is registered per PS rank from safetensors files or from in-memory `named_tensors` (`register_checkpoint`, ps.py:307-380).
- **Each rank owns a disjoint shard.** The example splits files or tensor names round-robin by rank (examples/update.py:52-75).
- **Pinned host memory:** registered tensors are copied into pinned CPU buckets of ≥ 4 GiB each, sorted by name, with 256-byte alignment (`_ALIGN_SIZE = 256`, pin_memory.py:23).
  - Normal path: pin_memory.py:277-362 (`torch.empty(size, pin_memory=True)`, :323; the copy, :326-327). 32 threads are used for allocation and copy (:329).
- **In-place pinning for /dev/shm:** safetensors files under `/dev/shm/` are pinned with `cudaHostRegister` and **deleted after pinning** (warning at ps.py:318; pin_memory.py:193-272, `cudaHostRegister` at :211, removal comment at :257).
- **Shared memory pool for repeated RL updates:** `use_shared_memory_pool=True` reuses one fixed-shape pinned pool, one user at a time (ps.py:339-360; reuse at pin_memory.py:312-321). This is the knob that avoids re-pinning every step.
- **Metadata:**
  - `ParameterMeta{name, dtype, shape, aligned_size}` (data_types.py:71-75);
  - per-buffer `MemoryBufferMetas{metas, ptr, size}` (:90-93);
  - per-rank `MemoryBufferMetaList{p2p_store_addr, memory_buffer_metas_list, rdma_device}` (:103-106);
  - `DataToGather` adds `host_ip` and `device_uuid` (:109-111).
- **`gather_metas`:** an `all_gather_object` builds `_current_global_parameter_metas: {owner_rank → buffers}` plus host, device-UUID and RDMA topology (ps.py:464-527).
- **Metas are portable:** they can be exported (`get_metas`, ps.py:294) and loaded into a new instance (`load_metas`, ps.py:297-305). The example writes them atomically to a file or serves them over HTTP (examples/update.py:116-128, 142-170). This is the **elasticity hook**: a new instance learns where every shard lives without the trainer.

### A.3 Broadcast vs P2P
`update(checkpoint_name, req_func, ranks=None)` (ps.py:606-657):
- **`ranks` empty → Broadcast.** Every PS rank takes part; "the fastest way … especially in colocated architecture" (docstring ps.py:622-625).
  - In `_gen_h2d_buckets` the **owner rank is also the sender** ("if ranks is empty, assign the owner_rank as receiver_rank, this is used for colocate architecture", ps.py:101-103).
  - Each owner copies its shard H2D and `dist.broadcast`s it to everyone (ps.py:927).
- **`ranks` set → P2P** into a subgroup (`dist.new_group(ranks)`, ps.py:636). This is for instances added while others keep serving [README:17]. Non-members return immediately (ps.py:836-837).
  - Receivers **RDMA-read** buckets directly from the owners' pinned CPU memory through Mooncake `TransferEngine` (`P2PStore`, p2p_store.py:11-40; `batch_transfer_sync_read`, ps.py:749-750, p2p_store.py:73-77).
  - Receivers then broadcast inside the subgroup.
- **P2P bucket assignment:** `_assign_receiver_ranks` (ps.py:108-163) selects one receiver per local RDMA device group (the minimum rank, :135-137). It interleaves owner buckets across RDMA devices (:139-146) and assigns round-robin, never using the same owner RDMA device twice in one round (:150-161), so every NIC stays busy [README:35-37].
- P2P is unsupported on XPU (ps.py:817-825).

### A.4 Bucketing and the pipeline (H2D → broadcast → reload)
**Bucket size** (`_detect_bucket_size`, ps.py:669-719):
- `all_reduce(MIN)` of `free_mem × PS_MEM_FRACTION` (default 0.9, ps.py:208) across ranks (:677-692).
- If the largest tensor ≤ free/3, it uses an extra **H2D staging buffer** so all ranks' H2D runs in parallel, and bucket = free/3 (:698-703).
- Otherwise it falls back to `disable_h2d_buffer` with bucket = free/2; bandwidth is then "limited by the h2d of a single machine" (:704-715).
- Bucket = `min(max(PS_MAX_BUCKET_SIZE_GB (default 8), max_tensor), free)` (:716-717).

**Buffers** (ps.py:850-870):
- an optional `h2d_buffer` of one bucket;
- a **double-size IPC `buffer` (2 × bucket)** exported once to the inference worker via CUDA IPC (`reduce_tensor`, ipc_handler.py:62-72; `ipc_handler.export`, ps.py:870).

**Protocol per bucket `gidx`** (ps.py:892-942), a ZMQ REQ (PS) ↔ REP (worker) lockstep:
1. **H2D:** copy this rank's next bucket from pinned memory into `h2d_buffer` (`_copy_to_buffer`, non-blocking copies then `synchronize`, ps.py:721-751, 893-899).
2. **Stage:** copy into half `gidx % 2` of the IPC buffer (`start = gidx % 2 * bucket_size`, ps.py:913-926).
3. **Broadcast:** `dist.broadcast(buffer_b, src=owner)` over the PS group. The bytes land in the IPC buffer, which the colocated vLLM worker already maps (ps.py:927).
4. **Wait for the previous reload:** `socket.recv()` returns the worker's ack for the previous message. Errors are all-reduced into `ret_code`, and any rank failing aborts everyone (ps.py:928-940).
5. **Reload:** send the bucket's tensor metadata (name, dtype, shape, offset) (ps.py:941). The worker views the tensors out of the shared buffer and calls `model.load_weights` (worker.py:39-51, 108-117).

- **Why this pipelines:** the worker's `load_weights` of bucket *i* (half *i % 2*) overlaps with H2D and broadcast of bucket *i+1* into the other half. The README calls the full three-stage overlap "not implemented" (README.md:229) and says it falls back to serial execution when memory is short (README.md:31).
- **Teardown:** two `None` messages. The first releases the IPC handle on the worker; the second runs `post_hook` = vLLM `process_weights_after_loading`, which handles quantized layers and MTP drafter weights (worker.py:78-107, 214-223; ps.py:944-969). A final barrier, and the process group is destroyed when `auto_pg` is set (ps.py:647-652, 970-977).

### A.5 How it plugs into vLLM and SGLang
- **vLLM:**
  - Start with `--worker-extension-cls checkpoint_engine.worker.VllmColocateWorkerExtension` and `VLLM_SERVER_DEV_MODE=1` (README.md:135-140).
  - The PS triggers `update_weights_from_ipc(zmq_handles)` on every worker through vLLM's dev-mode `/collective_rpc` HTTP endpoint (`request_inference_to_update`, api.py:15-43; `req_inference`, examples/update.py:78-94).
  - The extension resolves its ZMQ socket by **physical GPU UUID** (worker.py:150-162, 225-231; PS socket naming `ipc://@checkpoint-engine-{uuid}-{counter}.sock`, ps.py:659-667).
  - Loading also updates an MTP or speculative drafter model (worker.py:204-212).
- **SGLang:** `--load-format dummy --wait-for-initial-weights` plus `python -m sglang.srt.checkpoint_engine.update` [README:183-200]. The integration lives in SGLang, not in this repo.
- **Optional NCCL backend:** `vllm_nccl`, i.e. vLLM's `StatelessProcessGroup` + `PyNcclCommunicator` instead of torch.distributed (distributed/vllm_nccl.py:7-14, 107-163; selector `use_backend`, distributed/base.py:221-242).
- **Standalone server:** `python -m checkpoint_engine` over a Unix socket (`__main__.py`, api.py:46-108). Routes: register files, gather-metas, get/load metas, `update` with `ranks`, `update_url`, `inference_group_ranks`.
- **FP8:** vLLM needs a patch (`patches/vllm_fp8.patch`; README.md:176-181).

### A.6 What it does NOT do (searched `pause|abort|sleep_mode|wake_up|version|reset_prefix` in `checkpoint_engine/`; no hits beyond unrelated strings)
- **No in-flight request handling.** It does not pause, drain or abort the engine. Whether requests see a mid-update model depends on how vLLM schedules `collective_rpc` against decode steps (inference).
- **No prefix-cache reset** after an update. The caller must do it, or KV computed under the old weights is reused.
- **No policy-version bookkeeping.** `checkpoint_name` is the only identity.
- **No trainer integration.** The trainer must produce full (unsharded, HF-named) tensors on CPU per PS rank (`named_tensors`). Resharding to the inference TP layout is delegated to vLLM `load_weights`.
- **Tests need ≥ 2 GPUs** (`assert world_size >= 2`, tests/test_update.py:263-264, 295-296).

### A.7 Numbers [README benchmark, vLLM 0.10.2rc1, README.md:43-50]

| Model | Devices | GatherMetas | Broadcast | P2P |
|---|---|---|---|---|
| GLM-4.5-Air BF16 | 8×H800 TP8 | 0.12 s | 3.47 s (3.02 GiB bucket) | 4.12 s |
| Qwen3-235B-A22B BF16 | 8×H800 TP8 | 0.33 s | 6.22 s | 7.10 s |
| DeepSeek-V3.1 FP8 | 16×H20 TP16 | 1.17 s | 10.19 s | 11.80 s |
| Kimi-K2 (1T) FP8 | 16×H20 TP16 | 1.33 s | 14.36 s | 17.49 s |
| Kimi-K2 (1T) FP8 | 256×H20 (16 × TP16) | 1.22 s | 16.04 s (8 GiB) | 16.75 s |

The P2P column updates ≤ 16 GPUs. GPUs are bound to NUMA nodes (README.md:58-59). Cross-check: the survey cites Ant ASystem's P2P swap of 1T parameters in 10 s (survey §3.1).

### A.8 How it could serve Crucible
- **Single node, one GPU (the exit gate): don't use it.**
  - `world_size = 1` makes the broadcast a no-op. The pinned-host round trip (GPU → CPU `named_tensors` → pinned → H2D → IPC buffer) is strictly slower than what TRL colocate does: in-process `load_weights` per parameter with no copy across processes (trl/generation/vllm_generation.py:509-511). It is also slower than vLLM's native CUDA-IPC weight-transfer engine.
  - It also needs GPU memory for 2–3 buckets beside vLLM and the trainer (ps.py:850-863), which is the scarce resource on a 32 GB card.
- **Single node, multi-GPU (lab or Runpod, trainer on GPU 0, vLLM TP/DP on GPUs 1..N):**
  - Usable as the trainer → vLLM sync: the trainer rank-0 gathers a full state dict to CPU, and each PS rank registers `named_tensors` into the shared pool.
  - The real benefit is (a) one bucketed, double-buffered transfer instead of TRL/OpenRLHF's one RPC plus broadcast per parameter, and (b) **elastic join** of a restarted vLLM instance via `load_metas` + P2P.
  - The price is NUMA-pinned host RAM equal to the model size per node, plus vLLM dev mode.
- **Multi-node disaggregated (later M12):**
  - This is where it is designed to work. Trainer nodes dump weights to `/dev/shm` safetensors, which are pinned in place (pin_memory.py:193-272, 380-400). Each inference node runs `python -m checkpoint_engine` (api.py) and receives a broadcast. New nodes join via P2P from the existing nodes' pinned memory without stalling them.
  - Crucible must add what it lacks:
    1. a **policy version** = `checkpoint_name` stamped into every `Turn.policy_version` (`crucible.rollout/v1`; rollout.py `Turn.policy_version`);
    2. a **pause/keep → update → reset_prefix_cache → resume** wrapper around `ps.update`, the same pattern TRL AsyncGRPO uses (§C.6) and OpenRLHF partial rollout uses (§B.4);
    3. NCCL/RDMA and NUMA configuration.
- **Recommendation:** PATTERN now (`staleness.py`/`topology.py` design: the bucket plan, the double buffer, the ZMQ ack state machine with all-reduced error codes, metas export for elasticity). DEPEND (MIT) when M12 goes multi-node. Do not PORT: it is small, but vLLM-version-coupled (`collective_rpc`, `process_weights_after_loading`).

---

## B. OpenRLHF (0.11.3)

> **Flag names changed.** The CLI is now dotted and hierarchical (`hierarchize()`, openrlhf/utils/config.py:12-26), with no aliases. `--colocate_all_models`, `--async_train`, `--micro_train_batch_size`, `--vllm_gpu_memory_utilization`, `--enable_vllm_is_correction` and `LLMRayActor` no longer exist. Old names survive only in help strings (train_ppo_ray.py:248, 370, 712). The vLLM actor is `RolloutRayActor`.

### B.1 Ray topology
- **Entry:** `train()` (openrlhf/cli/train_ppo_ray.py:19-194).
- **Groups** (`RayActorGroup`, trainer/ray/launcher.py:202-373):
  - `PolicyModelActor` (train_ppo_ray.py:84-91);
  - `ReferenceModelActor` only if `--algo.kl.init_coef > 0` (:93-103);
  - `CriticModelActor` only for `gae`; any other estimator forces the critic off (:119-129, 644-645);
  - `RewardModelActor` only without `--reward.remote_url`; `--train.agent_func_path` forces `remote_url="agent"` (:132-142, 636-637);
  - vLLM `RolloutRayActor`s via `create_vllm_engines` (:66-82; vllm_engine.py:209-311);
  - one controller actor, `PPOTrainer` or `PPOTrainerAsync` (:145-148).
- **Colocation:**
  - `--train.colocate_actor_ref` / `--train.colocate_all` make one PACK placement group of `{"GPU":1,"CPU":1}` × actor GPUs (train_ppo_ray.py:41-50).
  - Each DeepSpeed actor in a shared group takes `num_gpus=0.2` (:89, 99, 125, 138), and vLLM with TP = 1 also takes 0.2 (vllm_engine.py:241-245). Up to five tenants share a GPU in Ray's accounting.
  - **vLLM shares the group only when `colocate_all and not async_enable`** (train_ppo_ray.py:75). Otherwise it builds its own PACK group of full-GPU bundles (vllm_engine.py:247-251).
  - Sync colocate asserts actor GPUs == engines × TP (train_ppo_ray.py:56-64).
- **Sleep mode:**
  - `--vllm.enable_sleep` maps to vLLM `enable_sleep_mode`. Engines sleep right after creation (vllm_engine.py:279, 308-309), and `sleep(level=1)` / `wake_up(tags)` are at vllm_engine.py:136-151.
  - The generator wakes everything before generation and sleeps after (samples_generator.py:106-120). The broadcast wakes only `["weights"]` (ppo_trainer.py:311-319).
  - Forced off without `colocate_all` and asserted incompatible with async (train_ppo_ray.py:707-715).
  - `--ds.enable_sleep` offloads DeepSpeed states around each `fit` (ppo_trainer.py:274-286; deepspeed_utils.py:151-189).

### B.2 Control flow: sync, async, partial rollout
- **Sync** (`PPOTrainer.fit`, ppo_trainer.py:503-579): `generate_samples` → `train_step` (= `make_experience_batch` → replay buffer → `ppo_train` → `broadcast_to_vllm`, :213-264).
  - Generation blocks on every ref (samples_generator.py:161-191), so the batch is finished before training.
  - Strictly on-policy apart from the usual PPO minibatch epochs.
- **Async** (`--train.async_enable`, train_ppo_ray.py:293; `PPOTrainerAsync`, ppo_trainer_async.py:268-353):
  - A generator actor and a trainer actor connected by a Ray `Queue(maxsize=async_queue_size)`, with a token pool `rollout_slots` for backpressure (ppo_trainer_async.py:292-297).
  - The generator takes a slot, runs one whole `generate_samples` batch and enqueues it (:100-164).
  - The trainer pops a batch and **returns the slot before `train_step`** (:202-232), so batch *k+1* is generated under weights *w_k* while batch *k* trains. That is one-step-off-policy at the default `async_queue_size=1` (train_ppo_ray.py:294).
- **Weight sync vs generation without partial rollout:** the generator holds a `VLLMLock` actor for the whole batch (ppo_trainer_async.py:128-140), and `broadcast_to_vllm` takes the same lock (:255-265). A sync therefore waits for the batch to finish, and every sample in a batch comes from one weight version.
- **Partial rollout exists:** `--train.partial_rollout_enable` (train_ppo_ray.py:295-302), which requires async (:717-718).
  - The generator stops taking the lock (ppo_trainer_async.py:128, 139). The trainer runs `pause_generation` → broadcast → `resume_generation` (:258-264).
  - The engine calls `self.llm.pause_generation(mode="keep")` (vllm_engine.py:130-134), so in-flight requests are **kept and resumed**, never aborted.
  - The help text says it directly: "In-flight samples may contain tokens from both old and new weights" (train_ppo_ray.py:299-301).
- **Staleness and version tracking: none.** grep for `stale|policy_version|model_version|weight_version` in `openrlhf/` finds nothing.
  - Lag is bounded only implicitly, by `async_queue_size` and by the over-generation buffer when `--rollout.vllm_generate_batch_size > --rollout.batch_size`, which is async-only (samples_generator.py:89-138; train_ppo_ray.py:740-744).
  - `old_action_log_probs` are recomputed by the trainer at experience time (experience_maker.py:154-158). In async mode they come from the trainer's current weights, not the behaviour policy. Only the IS correction against `rollout_log_probs` (§B.6) sees the true sampler.

### B.3 vLLM engines, dispatch, batching
- **Engine:** always `AsyncLLMEngine` (vllm_engine.py:63-64), with `worker_extension_cls = openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap` and bf16 (:264-280).
  - With IS on: `logprobs_mode="processed_logprobs"`, `max_logprobs=1` (train_ppo_ray.py:78; vllm_engine.py:293-298).
  - Generation is token-in (`TokensPrompt`, vllm_engine.py:153-177). n samples are n independent requests through `asyncio.gather`, not `SamplingParams.n` (:183-206).
- **Dispatch granularity: per prompt group.** `_dispatch_prompts_to_vllm` (samples_generator.py:195-241) snapshots `get_num_unfinished_requests` per engine and assigns each prompt with all its `n_samples` to the least-loaded engine through a min-heap. It makes **one Ray call per prompt** (`generate_responses.remote(..., num_samples=n_samples)`).
  - The whole group lands on one engine, which helps the prefix cache. Results are collected with `ray.wait(num_returns=1)`.
- **Batch knobs** (train_ppo_ray.py):

  | Flag | Default | Line |
  |---|---|---|
  | `--rollout.batch_size` (prompts/step) | 1024 | :392 |
  | `--rollout.vllm_generate_batch_size` | = rollout.batch_size | :393 |
  | `--rollout.n_samples_per_prompt` | 1 | :432 |
  | `--rollout.micro_batch_size` (experience forward chunk) | 1 | :396 |
  | `--train.micro_batch_size` (per GPU) | 1 | :412 |
  | `--train.batch_size` (global) | 128 | :413 |

  - Gradient accumulation is derived in openrlhf/utils/deepspeed/deepspeed.py:110-116.
- **Packing and dynamic batching:**
  - `--ds.packing_samples` (:375) needs flash-attn and vLLM (:699-705).
  - `--train.dynamic_batch_enable` + `--{train,rollout}.max_tokens_per_gpu` (:378-380) give token-budgeted micro-batches (replay_buffer.py:91-177).
  - Sequences are balanced by length (seqlen_balancing; experience_maker.py:49-69), and long and short samples are interleaved across DP ranks (`balance_experiences`, experience.py:250-283).

### B.4 Weight sync
- **NCCL group:**
  - A stateless group between DeepSpeed rank 0 and every vLLM rank: `world_size = engines × TP + 1`, `rank_offset = i × TP + 1` (ppo_actor.py:108-151).
  - Built by vLLM `StatelessProcessGroup` + `PyNcclCommunicator` (openrlhf/utils/distributed_util.py:13-26); worker side in vllm_worker_wrap.py:2-30.
  - Optional Ray collective: `--vllm.sync_with_ray` (train_ppo_ray.py:241).
- **Per-parameter, not bucketed:** `broadcast_to_vllm` (ppo_actor.py:409-490) loops over `named_parameters`. For each parameter it fires `update_weight.remote(name, dtype, shape)` on every engine, broadcasts, then `ray.get(refs)` (:421-437). That is one round trip per tensor.
  - ZeRO-3 is gathered one parameter at a time (`GatheredParameters`, :468-472, 484).
- **CUDA IPC:** used **only for sync colocate**: `use_cuda_ipc = backend == "nccl" and colocate_all and not async_enable` (ppo_actor.py:101).
  - Every rank clones the parameter, calls `reduce_tensor`, and the handles are keyed by physical GPU UUID and all-gathered (`_handle_cuda_ipc`, :439-466).
  - The worker rebuilds the tensor on its own device and calls `load_weights` (vllm_worker_wrap.py:52-70).
- **Prefix cache:** with `--vllm.enable_prefix_caching` it is reset at the start of the broadcast and awaited at the end (ppo_actor.py:410-415, 487-488). The return value is ignored.
  - (inference) Under partial rollout the paused "keep" requests still hold KV blocks, so vLLM may refuse the reset silently.
- **In-flight requests:**

  | Mode | Behaviour |
  |---|---|
  | sync | none in flight |
  | async | lock waits for the batch |
  | async + partial | pause(keep) → swap → resume; the trajectory mixes versions, unrecorded |

### B.5 Agent / multi-turn API
- **Plugging in:** `--train.agent_func_path <file.py>` must define `AgentExecutor(AgentExecutorBase)` (vllm_engine.py:19-30, 56-59). The default is `SingleTurnAgentExecutor`. Agent rollouts **run inside each `RolloutRayActor`**, next to the engine.
- **Interfaces** (openrlhf/utils/agent.py):
  - `AgentExecutorBase.execute(prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine, images=None)` (:13-16);
  - `AgentInstanceBase` with `async reset(states)` and `async step(state_dict)` (:19-29).
- **`MultiTurnAgentExecutor.execute`** (:32-185) is **token-in-token-out**:
  - It tokenizes the initial observation once. Each turn it calls `llm_engine.generate(current_obs_tokens)` (:100-103), keeps `outputs[0].token_ids` verbatim, records `action_ranges` (:104-111), and calls `step()`.
  - It tokenizes **only** the environment feedback and appends `current_obs_tokens + action_tokens + feedback_tokens` (:145-154).
  - Rollout logprobs: 0 for observation tokens, the sampled logprob for actions (:86-89, 157-162).
  - Exceptions: a left-truncated initial prompt is re-decoded (:75-77), and feedback tokenized in isolation can tokenize differently at the boundary.
- **Reward:** `step` returns a `rewards` tensor; per-turn rewards are **summed to one scalar** (:122) and placed on the last action token (models/utils.py:131-157).
  - There is **no per-turn credit channel**, so M6 turn advantages would need a fork.
  - Remote rewards: an HTTP RM server or a local `reward_func(queries, prompts, labels)` .py (agent.py:197-205, 308-365).
- **Example bugs:**
  - examples/python/agent_func_openai_server_executor.py:285 lacks the `images` parameter that `generate_responses` always passes (vllm_engine.py:195-203).
  - agent_func_gem_multiturn.py:49 declares `async def __init__`.

### B.6 Mismatch corrections, losses, advantages
- **IS correction** (train_ppo_ray.py:256-290):
  - `--algo.advantage.is_correction_level {off,token,seq}` (default off);
  - `--algo.advantage.is_correction_mode {mask,clip}`;
  - `--algo.advantage.is_correction_gating {ratio,binary_kl,tv}`;
  - `--algo.advantage.is_correction_threshold LOW HIGH` (default [0.5, 5.0]).
  - The comment at :258-259 maps them: TIS = token + clip; IcePop = token + mask; seq-mask-TIS = seq + mask; binary_kl = KPop-style.
  - Implementation in `PolicyLoss` (openrlhf/models/loss.py:214-247): `is_log_ratio = old_log_probs − rollout_log_probs`, then gate, then clip or mask, with `vllm_kl` and `is_filter_ratio` logged. It applies **only to `ppo`** (:216).
- **Losses:**
  - `--actor.policy_loss_type {ppo,gspo}` (train_ppo_ray.py:439). GSPO uses the sequence-mean log-ratio against `rollout_log_probs` when IS is on (loss.py:189-196).
  - Asymmetric clip `--actor.eps_clip_low_high` (:407); dual-clip `--actor.dual_clip` (:408).
  - `--actor.loss_agg_mode {token-mean, seq-mean-token-mean}` (:440).
  - **No CISPO** (grep).
- **Advantage estimators** (`--algo.advantage.estimator`, train_ppo_ray.py:514-522; experience_maker.py:268-317):
  - `gae` (with critic), `reinforce`, `rloo`, `reinforce_baseline`, `group_norm` (= GRPO), `dr_grpo` (mean only), `flash_reinforce`.
  - DAPO overlong penalty and ProRL stop penalty (:532-542).
  - **Dynamic filtering** (`--algo.dynamic_filtering_*`, :599-605): drop groups whose mean score is outside a range and replace the prompt (samples_generator.py:168-191).

### B.7 Single GPU
- **No 1-GPU example or script** (grep of examples/, README, docs for `num_gpus_per_node 1`, "single gpu", "minimum"). [README:771] recommends 8×A100-80GB.
- **Sync colocate with one engine** (inference from code, not run):
  - all `*.num_nodes 1 / num_gpus_per_node 1` (defaults are 8), `--train.colocate_all --vllm.num_engines 1`;
  - a non-GAE estimator, `reward_func.py`, `kl.init_coef 0`;
  - `--vllm.enable_sleep --ds.enable_sleep`.
  - DeepSpeed + Ray + vLLM on one card is heavy for a 32 GB GPU.
- **Async cannot run on one GPU:** with `async_enable`, vLLM gets its own full-GPU bundle (train_ppo_ray.py:75; vllm_engine.py:247-251) on top of the actor's bundle (train_ppo_ray.py:48), and the PACK group cannot be satisfied.

---

## C. TRL (1.16.0.dev0)

### C.1 Trainers and vLLM modes
- **`GRPOTrainer`** (trl/trainer/grpo_trainer.py) with `use_vllm=True` and `vllm_mode ∈ {"colocate" (default), "server"}` (grpo_config.py:582-598).
- **Colocate** (vllm_generation.py:311-368):
  - an **in-process `vllm.LLM`** with `distributed_executor_backend="external_launcher"`, `gpu_memory_utilization = vllm_gpu_memory_utilization` (default **0.3**, grpo_config.py:651-658), `max_num_batched_tokens=4096`;
  - **`logprobs_mode="processed_logprobs"`** "so temperature scaling/logit tweaking affects the TIS log probs" (vllm_generation.py:347-363);
  - optional sleep mode: `vllm_enable_sleep_mode` (grpo_config.py:607-613) puts the engine at `sleep(level=2)` after init and after every generate (vllm_generation.py:365-368, 750-752). Level 2 discards the weights, so they are re-pushed before the next generation (:586-590).
  - **[doc]** "Colocate mode (1 GPU, recommended) … requiring only a single GPU" (docs/source/openenv.md:331-337).
- **Server:**
  - `trl vllm-serve` or `vllm serve` with `VLLM_SERVER_DEV_MODE=1 --weight-transfer-config '{"backend":"nccl"}' --logprobs-mode processed_logprobs` (docs/source/openenv.md:343-353). `VLLMClient` talks to it (vllm_generation.py:300-309).
  - **[doc]** "Make sure that the server is using different GPUs than the trainer, otherwise you may run into NCCL errors" (docs/source/grpo_trainer.md:262).
- **`AsyncGRPOTrainer`** (trl/experimental/async_grpo/, §C.6) is server-only on **separate GPUs** (docs/source/async_grpo_trainer.md:65).

### C.2 How weights reach vLLM
`VLLMGeneration.sync_weights` (vllm_generation.py:481-517):
- **Colocate:** `for name, param in _iter_named_params(): llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights([(name, param)])` (:509-511).
  - Same process, same GPU, no IPC and no copy beyond `load_weights`.
  - `_iter_named_params` gathers one parameter at a time under ZeRO-3 or FSDP and merges and unmerges PEFT adapters (:435-479).
  - With sleep mode it first calls `wake_up(tags=["weights"])` (:489-492).
- **Server:**
  - The metadata `(name, dtype, shape)` is collected once (:500-504). `VLLMClient.update_named_params` posts `/update_weights` on a thread while the trainer streams a **packed NCCL broadcast** (`packed_nccl_broadcast_producer`, i.e. vLLM's native NCCL weight-transfer engine) (vllm_client.py:790-828).
  - The group is set up by `init_communicator`: trainer = rank 0, workers offset by 1, `POST /init_weight_transfer_engine` (vllm_client.py:702-747).
- **Both modes then reset the prefix cache** (vllm_generation.py:513-517).
- **When:** once per optimizer step, lazily, at the next generation (`if global_step != _last_loaded_step: sync_weights()`, grpo_trainer.py:1729-1733, 2136-2139).
  - Generation is synchronous, so **no request is ever in flight during a sync** in `GRPOTrainer`.

### C.3 Batching granularity and (near) on-policy control flow
- **Generation batch** = `per_device_train_batch_size × num_processes × steps_per_generation`; `steps_per_generation` defaults to `gradient_accumulation_steps` (grpo_config.py:1084-1101).
  - The batch must hold whole groups of `num_generations` (default 8, :479-485). `auto_find_batch_size` is rejected for this reason (:1072-1080).
- **Buffering:** one generation call produces `steps_per_generation` slices that are reused for `num_iterations` (μ) passes (grpo_trainer.py:1491-1507; sampler at :1214-1274). Generation is **batch-level and synchronous**: the whole batch goes to `llm.generate` in one call (vllm_generation.py:731).
- **Off-policy within the buffer:** `old_per_token_logps` are recomputed when `gradient_accumulation_steps % generate_every != 0`, and **always** when vLLM IS correction is on (grpo_trainer.py:2571-2586).
- **Multi-turn tool loop: turn-synchronous across the batch** (`_tool_call_loop`, grpo_trainer.py:1887-2125).
  - Each iteration executes the tool calls of every rollout that asked for one, then regenerates **all of them in one batched call** (`num_generations=1`, :2067-2071).
  - A slow tool or long rollout stalls the whole batch. This is the "long tail" the survey warns about (survey §2, challenge 4).
  - Async tools run concurrently on a daemon event loop (:1938-1953). Sync tools run sequentially (:1912-1932).

### C.4 Multi-turn and external environments
- **`tools=`** plain callables, and **`environment_factory=`**: a class or a `{name: class}` dict (grpo_trainer.py:266-285, 302-322, 630-709).
  - One instance per rollout, pooled and reset (:648-654).
  - `reset(**row)` may return an observation, which is appended to the last user message.
  - Every public method becomes a tool; the schema comes from the docstring (:665-681).
  - An optional `get_reward()` makes the environment own the reward, sync or async (:683-707).
  - Requires transformers ≥ 5.2.0 (:611-616).
- **Token-in-token-out:** after a tool call the next prompt is **`prompt_ids + completion_ids + tool_suffix_ids`** (grpo_trainer.py:1970-1983, "using concatenated IDs, no re-tokenization", :2065).
  - `_get_tool_suffix_ids` renders the tool message against a dummy conversation, slicing at the EOS so that Qwen3's `<|im_end|>\n` is handled (:1835-1885).
  - **`tool_mask`** is 0 on tool-result tokens and 1 on model tokens; logprobs for tool tokens are 0 (:2087-2098).
  - A non-prefix-preserving chat template (e.g. Qwen3's thinking strip) is **replaced** by a training-safe template (:752-756).
- **`rollout_func=`** (experimental): `Callable[[prompts, trainer], dict]` that must return `prompt_ids`, `completion_ids`, `logprobs` (grpo_trainer.py:125-129, 258-265, 2134-2154).
  - Extra keys are forwarded to the reward functions. **`env_mask`** is used as `tool_mask` (:2197-2200).
  - The trainer syncs vLLM weights before calling it (:2135-2139). The function receives the per-process prompt slice and must return `num_generations` completions per prompt itself.
  - **[doc]** "use `rollout_func` when … external agent servers … own the generation loop" (docs/source/openenv.md:598-605).
- **OpenEnv:**
  - Echo, Wordle, Catch, multi-env examples (docs/source/openenv.md:58-135, 216-360). Each environment instance holds a WebSocket session. Server concurrency must be ≥ `generation_batch_size`, set via `SUPPORTS_CONCURRENT_SESSIONS` and `max_concurrent_envs` (:572-596).
  - **Harbor** in-process (`trl/experimental/harbor/_env.py`: `HarborEnv.reset/_exec/reward`, :37-192; `HarborBashEnv.bash`, :210-228; example examples/grpo_harbor/grpo_harbor.py).
  - **OpenReward:** `trl/experimental/openreward/`.

### C.5 Mismatch correction and losses (GRPOConfig)
- **Rollout-mismatch IS** (grpo_config.py:920-957; grpo_trainer.py:2589-2636):
  - `vllm_importance_sampling_correction` (**default True**);
  - `vllm_importance_sampling_mode ∈ {token_truncate, token_mask, sequence_truncate, sequence_mask}` (default **`sequence_mask`**);
  - `vllm_importance_sampling_clip_max` (default 3.0) and `_clip_min`;
  - the ratio is exp(old − sampling) over `completion_mask × tool_mask`; NaN logprobs give ratio 1 (:2589-2603).
  - These cover **TIS** (token_truncate), **IcePop-style masking** (token_mask) and **MIS-style sequence masking**.
  - Logged: `sampling/sampling_logp_difference/{mean,max}` (= Crucible's `rollout_probs_diff`) and `sampling/importance_sampling_ratio/{min,mean,max}` (:2789-2825).
- **Other knobs:**
  - `off_policy_mask_threshold`: DeepSeek-V3.2 sequence masking of negative-advantage, high-KL sequences (grpo_config.py:958-965; `get_off_policy_mask`, grpo_trainer.py:2874).
  - `use_bias_correction_kl` (:967-978).
  - **`cast_lm_head_to_fp32`**: the FP32 LM head switch (grpo_config.py:53; grpo_trainer.py:989-1024).
- **`loss_type`** (grpo_config.py:796-830; grpo_trainer.py:3038-3110):
  - `grpo`, `dr_grpo`, **`dapo` (default)**, `bnpo`, `cispo` (clamped detached ratio × A × logp, :3038-3040), `sapo`, `luspo`, `vespo`.
  - **GSPO is `importance_sampling_level="sequence"`** (:755-763), not a loss type.
  - Clip-higher via `epsilon_high` (:700); two-sided via `delta` (:692).
  - `scale_rewards ∈ {group, batch, none}` (none = Dr.GRPO, :784-795).
  - `mask_truncated_completions` (DAPO overlong filtering, :831-838).
  - `top_entropy_quantile` (:861); entropy bonus and adaptive entropy (:871-911); `beta` default 0, so no reference model is loaded (:676-682).
- **Single GPU:** every loss above runs on one GPU. They are just loss formulas; `dapo` normalizes over the "global accumulated batch", which on one process is the local batch.
- **What is missing for Crucible:**
  - no dynamic-sampling (zero-variance group drop) flag (grep `dynamic_sampling|filter_groups` in `trl/trainer/` → none);
  - no critic or GAE in GRPO (→ PPO lives elsewhere);
  - no per-turn advantages, since one advantage is broadcast per completion (but `rollout_func` + reward funcs can carry extra fields, §G).

### C.6 AsyncGRPOTrainer (experimental): the async pattern to copy
- **Topology** (docs/source/async_grpo_trainer.md:23-43):
  - a rollout worker in a **separate spawned process** with `CUDA_VISIBLE_DEVICES=""` streams requests to a vLLM server;
  - the trainer consumes from an `mp.Queue`;
  - **vLLM and trainer on separate GPUs** (:65).
- **Dispatch granularity: per rollout (conversation)** (async_rollout_worker.py:534-708).
  - `max_inflight_tasks` slots; each free slot launches one `_generate_one` asyncio task with its own environment instance from a pool.
  - Groups are assembled as rollouts finish, and a group is scored when all `num_generations` arrive (:681-700). Each turn is one `/v1/completions` call with `return_token_ids`, `logprobs=0` and a random per-request seed (:985-1010).
  - `max_inflight_tasks` defaults to `max_staleness × samples_per_step` (async_grpo_config.py:348-356; trainer :1252-1259).
- **Versioning:**
  - The version is a shared `mp.Value` (async_rollout_worker.py:1157, 1176-1186). It is **stamped per group when the group's first rollout is dispatched** (`model_version=self.model_version`, :618).
  - The trainer drops a sample if `current − sample.model_version > max_staleness` (default 4; async_grpo_trainer.py:362-366; config :358-363) and logs `sample/staleness_{mean,max}` and `sample/dropped_stale_total`.
- **In-flight handling at sync:**
  - **Merged-weights mode:** `POST /pause?mode=keep` ("Freeze in-flight requests and clear caches", experimental/async_grpo/vllm_client.py:133-137) → packed NCCL broadcast → `/resume`, then the version is bumped (async_grpo_trainer.py:1749-1784; weight_transfer.py:132-183). In-flight turns continue under the new weights, but the group keeps its **start** version, so staleness is measured from w₀ (conservative, like GLM's rule).
  - **LoRA mode:** each sync publishes adapter `{name}-v{N}`, and requests name the version they started with, so a rollout finishes under one policy. The server must hold `max_staleness+2` adapters (docs/source/async_grpo_trainer.md:117-139; async_grpo_trainer.py:891-901, 1803-1841).
- **Multi-turn tokenization is *not* TITO here:** each turn re-renders the whole message list (async_rollout_worker.py:876-941) and reconciles drift afterwards.
  - `_chain_to_sequences` classifies each turn boundary as **CLEAN** (prefix-extends → append), **REALIGN** (drift < `fork_threshold_tokens`, default 1024 → overwrite as context) or **FORK** (start a new training row) (async_rollout_worker.py:87-195; config :288-296).
  - This is the same failure Crucible calls `prefix_break` (rollout.py `Turn.prefix_break`). TRL forks instead of masking.
- **Loop-owning harness mode (Harbor via OpenEnv):** `HarnessRolloutWorker` reads a validated OpenEnv `TrainingTrace` (engine prompt/completion ids, behaviour logprobs, per-token `loss_mask`) captured by a proxy. "TRL does not re-tokenize the captured prompts" (docs/source/openenv.md:607-700; trl/experimental/async_grpo/openenv_harness.py).
  - Rewritten histories start separate rows, and `None` reward = unscorable, dropped from the baseline (:666-692).
  - **This is Crucible's black-box design in TRL form.**
- **Batching on the trainer side:** samples are packed into token-budgeted rows balanced by Σ Lᵢ² across DP ranks (`token_budget`, async_grpo_config.py:334-346; doc :147-218).
- **Metrics worth copying verbatim into `monitors.py`:** `perf/rollout_wait_s` vs `rollout/backpressure_s` (which side is the bottleneck), `sample/time_in_queue_s`, `perf/weight_sync_{pause,barrier,transfer}_s`, `rollout/fork_frac`, `tools/unknown_name_total`, `tools/loop_exhausted_frac` (doc :220-330).

### C.7 Exact recipe: one RTX 5090 (32 GB), Qwen3-0.6B (or 1.7B + LoRA), GRPO, vLLM colocate, external multi-turn environment

**Why colocate:** server mode on one GPU puts two NCCL ranks on the same device. TRL warns against this (docs/source/grpo_trainer.md:262), and `AsyncGRPOTrainer` requires separate GPUs (async_grpo_trainer.md:65).

**Memory budget** (my arithmetic, to verify on the card):
- TRL loads string models in **fp32** unless `dtype` is given (`dtype` default `"float32"`, trl/trainer/utils.py:1326-1338), with bf16 autocast (`bf16` default True, grpo_config.py:410).
- **Qwen3-0.6B full fine-tune:** fp32 weights 2.4 GB + grads 2.4 GB + AdamW 4.8 GB ≈ 9.6 GB. Activations with gradient checkpointing (default True, :409) are a few GB at 4k tokens. vLLM at 0.30 × 32 GB ≈ 9.6 GB (bf16 weights 1.2 GB + KV cache). Fits.
- **Qwen3-1.7B full fine-tune in fp32 Adam** ≈ 27 GB before activations and vLLM: **does not fit**. Use LoRA (`peft_config`) or `optim="adamw_8bit"` + `model_init_kwargs={"dtype":"bfloat16"}`, with `vllm_enable_sleep_mode=True` as the fallback.

**Software:**
- `pip install "trl[vllm]"` (vLLM 0.21–0.31, pyproject.toml:84) and transformers ≥ 5.2.0 (environment_factory gate, grpo_trainer.py:611-616).
- **Blackwell (sm_120) needs a CUDA 12.8+ build of torch and vLLM.** Check that the installed `vllm` wheel supports sm_120 on the lab 5090 before the run (environment fact, not repo code).

```python
# crucible_trl_colocate.py   (env lives off-GPU: a Crucible World server or in-process CPU World)
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

CRUCIBLE_URL = "http://localhost:7801"   # captured from module scope (factory takes no args, openenv.md:143-147)

class CrucibleToolEnv:
    """One instance per rollout (pooled + reset by TRL). Public methods = tools; docstrings = schemas."""
    def __init__(self):
        self.client = WorldClient(CRUCIBLE_URL)          # Crucible's own client (HTTP/WebSocket); not TRL code
        self.episode = None
    def reset(self, task: str, seed: int, **kw) -> str:  # dataset columns arrive as kwargs (grpo_trainer.py:266-273)
        self.episode = self.client.start(task=task, seed=seed)
        return self.episode.observation                  # appended to the last user message
    def get_reward(self) -> float:                       # env owns the reward (grpo_trainer.py:683-707)
        return self.client.grade(self.episode).reward    # None-safe: TRL turns None into NaN and ignores it
    # one method per World action, typed + Google docstring, e.g.:
    def lookup_ticket(self, ticket_id: str) -> str:
        """Look up a ticket.

        Args:
            ticket_id: The ticket id.

        Returns:
            The ticket record.
        """
        return self.client.act(self.episode, "lookup_ticket", ticket_id=ticket_id)

ds = Dataset.from_list([{"prompt": [{"role": "user", "content": "Work the queue."}], "task": t, "seed": s}
                        for t, s in TASKS])           # >= 1 row per (task, seed) group

args = GRPOConfig(
    output_dir="runs/qwen3-0.6b-crucible",
    use_vllm=True, vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.30, vllm_max_model_length=8192,   # grpo_config.py:651-665
    vllm_enable_sleep_mode=False,            # set True if OOM (wake/sleep adds H2D per step, vllm_generation.py:489-492, 750-752)
    num_generations=8, per_device_train_batch_size=8, gradient_accumulation_steps=4,   # 4 groups/step
    max_completion_length=4096,              # TOTAL over all turns incl. tool tokens (openenv.md:203-214)
    max_tool_calling_iterations=16,
    loss_type="dapo", epsilon=0.2, epsilon_high=0.28, mask_truncated_completions=True,
    scale_rewards="group", beta=0.0,
    vllm_importance_sampling_correction=True, vllm_importance_sampling_mode="token_truncate",
    vllm_importance_sampling_clip_max=2.0,   # TIS; logs sampling_logp_difference = rollout_probs_diff
    cast_lm_head_to_fp32=False,              # flip to test the survey's "cheapest single fix"
    chat_template_kwargs={"enable_thinking": False},
    learning_rate=1e-6, logging_steps=1, bf16=True, gradient_checkpointing=True,
    log_completions=True,
)
trainer = GRPOTrainer(model="Qwen/Qwen3-0.6B", args=args, train_dataset=ds,
                      environment_factory=CrucibleToolEnv)   # reward_funcs defaults to None (grpo_trainer.py:305, 521)
trainer.train()
```

```bash
CUDA_VISIBLE_DEVICES=0 python crucible_trl_colocate.py   # single process; no accelerate config needed on 1 GPU
```

**Notes:**
- `reward_funcs` may be omitted when the environment defines `get_reward`; otherwise init raises "No reward source provided" (grpo_trainer.py:711-717).
- TRL already logs `frac_reward_zero_std` (grpo_trainer.py:2759), the zero-variance-group share DESIGN §10.3 tracks as a collapse warning. It does not drop those groups.
- Crucible Worlds expose task-specific tool sets, so generate one env class per World family (`type(...)`) and use the **dict form** of `environment_factory` with an `environment` column (grpo_trainer.py:637-646, 266-285).
- **For exact Crucible TITO and per-call credit, use `rollout_func` instead.**
  - Drive Crucible's own policy loop and generate each turn with `trainer.vllm_generation.generate(prompt_ids, None, num_generations=1)` (vllm_generation.py:550-574). That is token-id in, `(prompt_ids, completion_ids, logprobs, logprob_token_ids)` out.
  - Concatenate the deltas exactly like `_tool_call_loop`, and return `prompt_ids/completion_ids/logprobs/env_mask` plus extra fields (`rollout_id`, `policy_version`, `turn_spans`) for reward functions and the M12 bridge (grpo_trainer.py:2134-2154, 2197-2200).
  - The colocated engine has no HTTP endpoint, so Crucible's `LLMToolPolicy` needs a local adapter that calls `vllm_generation.generate` instead of an OpenAI URL.

---

## D. rLLM (0.3.0.pre; `verl==0.8.0` extra, pyproject.toml:77-80)

> **The old stack is gone.** `AgentExecutionEngine`, `AgentPPOTrainer`, `n_parallel_agents` and the `AsyncLLMServerManager` subclass are no longer defined. env_agent_mappings.py:6 says "After the cleanup of the Agent+Environment+AgentExecutionEngine stack, the env and agent maps are empty", with `ENV_CLASSES = {}` and `AGENT_CLASSES = {}` at :20-21. The legacy agent_class/env_class paths were removed (rllm/trainer/agent_trainer.py:22-24). The backends are `Literal["verl","tinker","fireworks"]` (rllm/trainer/unified_trainer.py:1009). There is no SkyRL backend (grep).

### D.1 Agent and environment abstractions
- **Data model** (rllm/types.py):
  - `Step` carries `prompt_ids`, `response_ids`, `logprobs`, `routing_matrices`, `model_output`, `advantage` (scalar or per-token) and **`weight_version`** "(async staleness)" (:100-136).
  - `model_post_init` back-fills the token fields from `ModelOutput` and asserts `len(response_ids) == len(logprobs)` (:149-167).
  - `Trajectory{uid, name, steps, reward, signals}` (:241-254); `Episode{id="task_id:rollout_idx", termination_reason, is_correct, trajectories, metrics}` (:317-338); `TrajectoryGroup{trajectories, group_id, weight_version}` (:384-401).
- **`BaseAgent`:** `update_from_env`, `update_from_model → Action`, `reset`, `chat_completions` (rllm/agents/agent.py:40-100).
- **`BaseEnv`:** `reset() → (obs, info)`, `step(action) → (obs, r, done, info)`, `from_dict`, `is_multithread_safe` (rllm/environments/base/base_env.py:5-79).
- **`AgentFlow`, the newer protocol:** `run`/`arun(task, AgentConfig{base_url, model, session_uid, sampling_params}) → Episode | Trajectory` (types.py:417-455).
  - Sync flows run in a thread pool (types.py:525-553).
  - `SandboxedAgentFlow` receives a `Sandbox{exec, upload_file, upload_dir, close}` (rllm/sandbox/sandboxed_flow.py:21-69; protocol.py:17-44).
  - **An AgentFlow needs only an OpenAI `base_url`. This is the black-box harness path.**

### D.2 Workflow API
- **`Workflow(rollout_engine, executor, timeout, gamma, reward_bonus_coeff)`** with abstract `async run(task, uid)` (rllm/workflows/workflow.py:34-67).
- **Termination:** `run_with_termination_handling` = `asyncio.wait_for(run, timeout)`, mapping to `TerminationReason` {MAX_PROMPT_LENGTH_EXCEEDED, MAX_RESPONSE_LENGTH_EXCEEDED, ENV_DONE, MAX_TURNS_EXCEEDED, TIMEOUT, UNKNOWN, ERROR} (:18-25, 80-105).
- **Assembly:** `commit()` and `collect_trajectories()` build the episode.
  - `postprocess_episode` sets trajectory reward = Σ step rewards, applies optional MC discounting with `gamma`, sets `is_correct = reward > 0`, and sets the termination reason (:107-257).
  - Reference loop `MultiTurnWorkflow.run`: env.reset → agent.update_from_env → LLM call (sticky `application_id=uid`) → update_from_model → env.step, run in the executor (rllm/workflows/multi_turn_workflow.py:34-56; timing_mixin.py:78-90).
- **`UnifiedWorkflowEngine`:** a pool of `n_parallel_tasks` workflow instances in an `asyncio.Queue` (default 256, base.yaml:36) plus a thread pool for blocking env calls (rllm/engine/unified_workflow_engine.py:73, 97-118).
  - `process_task_with_retry` retries only `ERROR` terminations up to `retry_limit` (:120-181).
  - `execute_tasks` makes one coroutine per (task, rollout) and consumes them with `as_completed` (:183-240).
- **`AgentFlowEngine`:** semaphore-bounded. Per attempt it opens a gateway session, runs the flow, fetches the gateway traces, enriches the episode, then evaluates (rllm/engine/agentflow_engine.py:377, 476-517, 547, 613-680).
  - Strict mode raises `EnrichMismatchError` on missing token ids so the rollout is retried (:175-182).

### D.3 Rollout engines and dispatch
- **`RolloutEngine.get_model_response`** stamps `result.weight_version = self.weight_version`, captured **before** the call (rllm/engine/rollout/rollout_engine.py:87-91). `ModelOutput` = {prompt_ids, completion_ids, logprobs, routing_matrices, finish_reason, weight_version} (:15-31).
- **`VerlEngine`:** one request per LLM call: `server_manager.generate(request_id=application_id, prompt_ids=...)` with `logprobs=1` (verl_engine.py:39-51, 83-89).
  - Load balancing is verl's `LLMServerClient`: least in-flight, sticky by `request_id` (rllm/trainer/verl/verl_backend.py:191-200, 332-346).
  - An aborted output raises (verl_engine.py:91-92).
- **`OpenAIEngine`:** with a tokenizer it calls `/completions` with token ids. **Without one it returns empty token ids and logprobs** (openai_engine.py:132-144, 160-257), which is exactly the case Crucible must mask.
- **Tinker and Fireworks:** remote token-in/token-out sampling, one sample per call (tinker_engine.py:325-329; fireworks_engine.py:43-51).
- **Wake/sleep are no-ops:** `RolloutEngine.wake_up/sleep` are `pass` (rollout_engine.py:108-112).

### D.4 TITO: what rLLM actually does
- **The default path re-renders and re-tokenizes the whole history every turn:** `prompt = chat_parser.parse(messages, ...)` then `tokenizer.encode(prompt)` (verl_engine.py:104-105; same in tinker_engine.py:403-411 and openai_engine.py:256). Completion ids and logprobs are exact from the server (verl_engine.py:135-136).
- **Prefix merge at training time:**
  - The verl transform merges consecutive steps when step *k*'s prompt starts with the previous prompt + completion (rllm/trainer/verl/transform.py:372-378). The result is one row `[A0, obs1, A1, …]`, mask 1 on actions and 0 on observations (:383-390).
  - **A prefix break emits a new row** (:394-398), logged as `batch/merge_compression_ratio` (:457-510). Steps without `prompt_ids` are dropped (:292-295).
  - Tinker does the same (rllm/trainer/tinker/transform.py:22, 42-128).
  - Same design as TRL AsyncGRPO's FORK. Crucible's `prefix_break` masks instead.
- **`TITOCompleter`** (delta tokenization, rllm/engine/rollout/completer.py:60-151) exists but **nothing uses it** (grep `Completer`). It has apparent bugs: a non-prefix delta still gets the previous tokens prepended (:98-112), and `prompt_ids` is not passed (:115), so the verl transform would drop the step.
- **`rllm-model-gateway`, the capture proxy for opaque agents:** `/sessions/{sid}/v1/...` injects `logprobs=True` and `return_token_ids=True` and pins the model and the session's sampling params (rllm-model-gateway/src/rllm_model_gateway/middleware.py:124-140).
  - It stores a `TraceRecord{prompt_token_ids, completion_token_ids, logprobs, routing_matrices, finish_reason, weight_version}` (models.py:9-31). The weight version comes from `POST /admin/weight_version` (proxy.py:137; server.py:403-413).
  - **`cumulative_token_mode`** makes it true TITO for black-box agents. `TokenAccumulator` keeps the previous prompt + completion ids per session and bridges to the next turn (token_accumulator.py:53-153). The proxy rewrites turn 2+ into `/v1/completions` with `prompt = token_ids` (proxy.py:151-185, 287-288) and resets on divergence (:156-160).
  - Session routing is `StickyLeastLoadedPolicy`, "Mirrors verl `AsyncLLMServerManager._choose_server()`" (session_router.py:70-99).
- **Dead config:** `filter_token_mismatch: True` (rllm/trainer/config/rllm/base.yaml:109) is read by no Python code (grep).

### D.5 How it trains
- **Unified `AgentTrainer(config, workflow_class | agent_flow + evaluator, backend=…)`** → `UnifiedTrainer(backend_cls)` (unified_trainer.py:979-1111). Backend protocol: `generate_episodes`, `transform_to_backend_batch`, `process_backend_batch`, `compute_advantages`, `update_policy`, `on_policy_updated` (rllm/trainer/backend_protocol.py:29-209).
- **Advantages are trajectory-level and broadcast:** `assert stepwise_advantage_mode == "broadcast"`, then every step gets the trajectory's scalar (rllm/trainer/algorithms/advantage.py:188, 228-232).
  - Estimators: grpo, reinforce, reinforce_plus_plus_baseline, prpo, rloo, plus custom ones via `register_rllm_adv_estimator` (advantage.py:25-136; config.py:241-253).
  - **`use_precomputed_advantage`** accepts per-token advantages (advantage.py:139-168, 199-203). This is the hook for M6 per-turn credit.
- **verl backend:**
  - Episodes become a DataProto with `response_mask` 0 on observations, rewards at the last token, optional `rollout_log_probs` (rllm/trainer/verl/transform.py:135-234).
  - `old_log_probs` are recomputed, or `bypass_mode` reuses the rollout logprobs. Optional TIS uses verl `compute_rollout_correction_weights` (verl_backend.py:627-676). Config: `rollout_correction.{bypass_mode,tis_mode,tis_cap}` (base.yaml:89-92).
  - Losses through verl `ppo_loss` with per-role `policy_loss_mode_override` (verl_backend.py:64-91, 745-825).
  - Weights move through verl's `CheckpointEngineManager` (verl_backend.py:202-208, 875-878, 891-895).
- **Tinker backend (remote, no local GPU):** loss map `{importance_sampling, ppo, cispo, dro, cross_entropy}` (rllm/trainer/tinker/tinker_policy_trainer.py:37-47). `bypass_mode: true` by default (tinker.yaml:74-76).
- **Filters:**
  - rejection sampling (`episode` mode; `group` raises NotImplementedError, rejection_sampling.py:181-182);
  - `filter_uniform_groups` (buffer.py:236-240);
  - compact filtering by termination reason (rllm/trainer/algorithms/transform.py:105-152).

### D.6 Control flow, staleness, in-flight handling
- **Default is synchronous and on-policy** (`_fit_on_policy`, unified_trainer.py:436-490): generate the batch, then train. With verl colocated, the replicas sleep after generation and weights are updated after the step (verl_backend.py:420-423, 875-878).
- **Fully async** (`async_training.enable`, base.yaml:170-178; `_fit_fully_async`, unified_trainer.py:585-627):
  - The generation loop dispatches **one group at a time** (task × `rollout.n` asyncio tasks), gated by `SyncCoordinator` (:629-667).
  - **Admission quota:** `max_rollout_quota = (1 + staleness_threshold) × trigger_parameter_sync_step × mini_batch_size`. The counter resets only at a weight sync, and in-flight groups carry into the new window (rllm/trainer/sync_coordinator.py:16-19, 59-101). "This guarantees zero staleness when staleness_threshold=0" (:26-28).
  - **Version per group = min over its steps' `weight_version`** (rllm/trainer/buffer.py:188, 258-259, 414-421). This is GLM's w₀ rule, the same as Crucible's `RolloutRecord.oldest_version()`.
  - `async/staleness_{mean,min,max}` are logged (unified_trainer.py:746-749). **There is no drop rule:** the quota is the bound.
- **Weight sync:** `_perform_weight_sync` (unified_trainer.py:812-827).
  - `partial_rollout=false`: `pause_generation()` + `wait_for_drain()`, so in-flight rollouts finish on the old weights.
  - `partial_rollout=true`: no pause. Resuming aborted requests is delegated to verl's `FullyAsyncLLMServerClient` (rllm/trainer/verl/async_agent_loop.py:10-31).
  - Then the version is bumped on the rollout engine and the gateway.
- **Experimental SGLang fully-async stack:**
  - `RolloutExecutor` caps `max_required_samples = required × (staleness_threshold+1) × trigger_step` (rllm/experimental/fully_async/rollout_executor.py:19-70, 342-371).
  - Pause = SGLang `/pause_generation` mode "abort" (:224-241; utils.py:218-233). `RolloutClient` re-submits `output.all_tokens()` after an abort and **stamps each chunk with its version** (client.py:59-124), i.e. partial rollout with per-span versions.
  - `ParameterSynchronizer`: pause → version → NCCL sync → resume (param_sync.py:120-178).

### D.7 Resources
- **No checked-in 1-GPU script** (grep `n_gpus_per_node=1`).
- Qwen3-0.6B examples use 8 GPUs (examples/countdown/train_countdown.sh:15, 58), or 4 + 4 for async with `staleness_threshold=4.0` and `partial_rollout=true` (examples/countdown/unified_trainer/train_countdown_unified_verl_async.sh:8, 46-57).
- **[doc]** a "Single-GPU Training" section exists (docs/guides/distributed-training.mdx:22-25).
- Tinker needs no local GPU but is a remote paid service, so it cannot use the lab 5090.

---

## E. Cross-cutting: granularity, sync vs async, staleness, in-flight requests

| Aspect | checkpoint-engine | OpenRLHF | TRL GRPOTrainer | TRL AsyncGRPO | rLLM |
|---|---|---|---|---|---|
| **Inference dispatch unit** | n/a (weight plane only) | one Ray call **per prompt group** (n requests on one engine), least-loaded heap (samples_generator.py:195-241) | **whole generation batch** in one `llm.generate` (vllm_generation.py:731); tool turns batched per iteration (grpo_trainer.py:2067-2071) | **per rollout** asyncio task, one `/v1/completions` per turn (async_rollout_worker.py:534-708, 985-1010) | **per request** (per LLM turn), sticky per trajectory (verl_engine.py:83-89; session_router.py:70-99) |
| **Scheduling unit** | n/a | batch (`generate_samples`) | batch | rollout (slot pool `max_inflight_tasks`) | sync: batch; async: **group** (task × n) (unified_trainer.py:645-661) |
| **Training consumption** | n/a | batch from queue | buffered generation batch, `steps_per_generation` slices × μ | samples from `mp.Queue`, token-budget packed rows | async: `mini_batch_size` groups (unified_trainer.py:669-743) |
| **Control flow** | synchronous update call | sync, or async (1 generator + 1 trainer actor, queue + slot tokens) | synchronous | async (separate process + server) | sync, or fully async (two asyncio loops + coordinator) |
| **Staleness bound** | none | implicit: `async_queue_size` (+ over-generation buffer); **no version tags** | ≤ μ·steps_per_generation micro-steps inside one buffer; old logps recomputed | **explicit drop**: `current − v_group > max_staleness` (async_grpo_trainer.py:362-366) | **admission quota** (sync_coordinator.py:16-19); no drop |
| **Version granularity** | `checkpoint_name` | none | n/a (on-policy per step) | per **group**, at dispatch (async_rollout_worker.py:618) | per **step**; group = min (buffer.py:414-421); per chunk in experimental (client.py:59-124) |
| **In-flight at sync** | not handled | sync: none; async: lock waits for batch; partial: `pause(mode="keep")` → swap → resume, **mixed versions unrecorded** (vllm_engine.py:130-134) | none in flight | merged: `/pause?mode=keep` → NCCL → `/resume` (in-flight continue under new weights; group keeps start version); LoRA: versioned adapters, no mixing | no partial: pause + **drain**; partial: verl abort + resume; experimental: SGLang abort + resubmit with per-chunk version |
| **Prefix-cache reset after sync** | no (caller) | yes if prefix caching on (ppo_actor.py:410-415, 487-488) | yes (vllm_generation.py:513-517) | inside vLLM `/pause` ("clear caches", vllm_client.py:133-137) | delegated to verl |
| **Transfer mechanism** | bucketed double-buffered NCCL broadcast into a CUDA-IPC buffer + ZMQ; P2P RDMA | per-param NCCL broadcast (stateless PyNccl) or CUDA IPC (sync colocate only) | colocate: in-process `load_weights`; server: vLLM packed NCCL engine | vLLM packed NCCL engine, or LoRA adapter files | verl `CheckpointEngineManager`; Tinker sampling-client swap |

**Takeaways for `staleness.py`:**
1. **Version every span at sampling time**, as rLLM does: `RolloutEngine` stamps `ModelOutput.weight_version` before the call (rollout_engine.py:87-91), and the gateway stamps every trace (proxy.py:137). Crucible's `Turn.policy_version` already has the slot (rollout.py).
2. **Compute staleness from the oldest span** (rLLM buffer.py:414-421 ≡ `RolloutRecord.oldest_version()`). TRL AsyncGRPO's per-group dispatch stamp is equivalent only because it never re-dispatches.
3. **Bound staleness twice:**
   - admission (rLLM quota formula, counter reset at sync, in-flight carryover; sync_coordinator.py:16-19, 93-101);
   - plus a hard drop at consumption (TRL AsyncGRPO async_grpo_trainer.py:362-366).
   OpenRLHF has neither, so do not copy its async loop as-is.
4. **Partial rollout with `pause(mode="keep")` mixes versions inside one completion.** Both OpenRLHF (train_ppo_ray.py:299-301) and TRL merged mode do it. Crucible must either:
   - (a) record the swap point: after `/resume`, tag the remainder of each in-flight request with the new version; this needs engine support (inference); or
   - (b) use abort + resubmit-with-prefix as rLLM's experimental stack does (client.py:59-124), which gives exact per-chunk versions at the cost of a re-prefill (MiMo's choice, survey §3.2).
   Option (b) fits Crucible's linear delta-token storage (rollout.py `Turn.prompt_is_delta`).
5. **Scheduling granularity:** rollout-level dispatch (TRL AsyncGRPO) or group-level admission (rLLM). Never batch-level, which OpenRLHF and TRL GRPOTrainer use. DeepSeek reported that batch-level scheduling made metrics oscillate and prompt-level scheduling stalled on group tails (survey §3.2). Per-sample dispatch with group assembly in a buffer is the converged answer, and both TRL AsyncGRPO and rLLM implement it.

---

## F. Comparison table

| | checkpoint-engine | OpenRLHF 0.11.3 | TRL 1.16.dev | rLLM 0.3.0.pre |
|---|---|---|---|---|
| **Role** | weight plane only | full trainer (Ray + DeepSpeed + vLLM) | full trainer (Accelerate/DeepSpeed/FSDP + vLLM) | agent/workflow layer over verl / Tinker / Fireworks |
| **License** | MIT | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| **1 GPU** | pointless (tests need ≥ 2) | sync colocate only, untested; async impossible (§B.7) | **yes: colocate, documented** (openenv.md:331-337) | no script; verl single-GPU [doc only]; Tinker remote |
| **Async** | — | yes (batch-level queue) + partial rollout | AsyncGRPO (≥ 2 GPUs) | yes (group-level, quota) + partial rollout |
| **Staleness control** | — | none | drop at `max_staleness` | admission quota + metrics |
| **Version tags** | name only | none | per group | per step (+ per chunk, experimental) |
| **Multi-turn API** | — | `AgentInstanceBase.step` inside the vLLM actor | `environment_factory` / `tools` / `rollout_func`; Harbor/OpenEnv loop-owning (async) | `Workflow`, `AgentFlow` (+ gateway), `BaseAgent`/`BaseEnv` |
| **TITO** | — | **yes** (append feedback tokens; agent.py:145-154) | **yes** in GRPOTrainer tool loop (grpo_trainer.py:1970-2065); **no** in AsyncGRPO (re-render + FORK) | re-render + training-time prefix merge (row split on break); gateway `cumulative_token_mode` = TITO for opaque agents |
| **Rollout logprobs** | — | `processed_logprobs`, logprobs = 1 | `processed_logprobs` (vllm_generation.py:361) | `logprobs=1`; gateway injects |
| **Mismatch correction** | — | token/seq × mask/clip × ratio/binary_kl/tv (PPO only) | token/seq × truncate/mask (default `sequence_mask`, C = 3), off-policy seq mask, FP32 LM head | verl TIS / bypass mode |
| **Objectives** | — | ppo, gspo, dual-clip; gae, rloo, group_norm, dr_grpo, reinforce(+baseline) | grpo, dr_grpo, **dapo** (default), bnpo, **cispo**, sapo, luspo, vespo; GSPO via sequence IS | grpo, rloo, reinforce(++), prpo, custom; Tinker: ppo, IS, cispo, dro |
| **Critic** | — | yes (gae) | no (GRPO) | via verl |
| **Per-turn credit** | — | no (Σ rewards → last token) | no (one advantage per completion) | `use_precomputed_advantage` (per token) |
| **Zero-variance group drop** | — | dynamic filtering | logs `frac_reward_zero_std` only | `filter_uniform_groups`, rejection sampling |

---

## G. Recommendations for Crucible M12

**Dependency rule** (memory "build on frameworks"): every module reuses agentenv or RL-Environments frameworks. All four repos are permissive (MIT / Apache-2.0), so PORT is legal with attribution and a NOTICE.

| Crucible piece | Verdict | Source (file:line) | Why |
|---|---|---|---|
| **Exit-gate trainer** (one colocated GRPO run on the 5090) | **DEPEND: TRL `GRPOTrainer`, `vllm_mode="colocate"`** | grpo_trainer.py:2127-2200 (`rollout_func`/tool loop), vllm_generation.py:311-368, 481-517 | only one of the four that is 1-GPU-native with TITO tool loops, IS correction, DAPO/CISPO/GSPO |
| `bridges/trl.py`: Crucible World → trainer | **DEPEND** via `rollout_func` (returns `prompt_ids`, `completion_ids`, `logprobs`, `env_mask` + extra fields) | grpo_trainer.py:125-129, 2134-2154, 2197-2200 | Crucible keeps its own loop, record and credit; TRL only optimizes |
| Delta-token concat for tool results | **PATTERN** (`_get_tool_suffix_ids` EOS-aligned slicing) | grpo_trainer.py:1835-1885 | the exact Qwen3 `<\|im_end\|>\n` boundary bug Crucible's `prefix_break` would otherwise hit |
| `staleness.py` admission | **PORT** (Apache-2.0, ~170 lines) `SyncCoordinator` | rllm/trainer/sync_coordinator.py:9-172 | quota formula, carry-over at sync, drain/pause, task-error propagation; framework-free asyncio |
| `staleness.py` drop + metrics | **PATTERN** | trl async_grpo_trainer.py:346-385 | drop `v_now − v₀ > τ`; `staleness_mean/max`, `dropped_stale_total`, `time_in_queue_s` |
| Group assembly from per-rollout streams | **PATTERN** | rllm/trainer/buffer.py:166-280; trl async_rollout_worker.py:534-708 | per-sample dispatch, score when the group completes, group version = min over steps |
| Prefix-break handling | **PATTERN, but keep Crucible's mask default.** Offer FORK as an option | trl async_rollout_worker.py:87-195; rllm/trainer/verl/transform.py:372-398 | both split rows; Crucible masks (rollout.py `valid_for_training`). A `fork` mode keeps data, with `fork_frac` monitored |
| `mismatch.py` | **PATTERN** (union of both menus) | trl grpo_trainer.py:2589-2636 (4 modes), 2789-2825 (metrics); openrlhf models/loss.py:214-247 (ratio/binary_kl/tv gating) | TRL lacks binary-KL (KPop); OpenRLHF lacks sequence truncate and DeepSeek off-policy masking. Write once in Crucible, test on synthetic ratios (exit gate) |
| `objectives.py` | **DEPEND on TRL** for grpo, dapo, dr_grpo, cispo, GSPO(seq); **PATTERN** OpenRLHF dual-clip, GAE | trl grpo_trainer.py:3038-3110; openrlhf experience_maker.py:343-387 | critic mode needs GAE, which TRL GRPO lacks; take it from OpenRLHF/verl later |
| Capture proxy for opaque harnesses (M10/M3) | **PATTERN or DEPEND** `rllm-model-gateway` (cumulative token mode, weight-version stamping) | rllm-model-gateway/src/rllm_model_gateway/{middleware.py:124-140, token_accumulator.py:53-153, proxy.py:137-300} | standalone (6 deps, no torch); same role as verifiers interception / OpenEnv capture already in ARCHITECTURE |
| Weight sync, 1 GPU | **DEPEND TRL colocate** (in-process `load_weights`) | vllm_generation.py:509-511 | zero-copy; no NCCL on a shared device |
| Weight sync, multi-GPU single node | **DEPEND** vLLM native NCCL weight-transfer engine (as TRL server/AsyncGRPO do) | trl generation/vllm_client.py:702-828; async_grpo/weight_transfer.py:80-183 | packed broadcast, no extra pinned host copy |
| Weight sync, multi-node / elastic | **DEPEND** checkpoint-engine (MIT) | ps.py:606-977; worker.py:134-231; api.py:46-108 | bucketed double buffer, P2P join from pinned memory, ~16 s for 1T on 256 GPUs [README] |
| Partial rollout | **PATTERN**: abort + resubmit with per-chunk versions | rllm/experimental/fully_async/client.py:59-124; rollout_executor.py:224-241 | exact `policy_version` per span; avoid `pause(mode="keep")` mixing (OpenRLHF train_ppo_ray.py:299-301) |
| Per-turn advantages → tokens | **PATTERN** rLLM `use_precomputed_advantage`; TRL: carry per-token advantages as an extra `rollout_func` field and override `_compute_loss` (subclass) | rllm/trainer/algorithms/advantage.py:139-168; trl grpo_trainer.py:2949 | M6 credit must reach tokens; none of the four ships turn-level credit natively |
| Monitors | **PATTERN** TRL AsyncGRPO metric set | docs/source/async_grpo_trainer.md:220-330 | wait vs backpressure, weight-sync phase times, fork/realign fractions, tool unknown-name rate |
| OpenRLHF as a runtime | **No DEPEND** for the MVP; PATTERN only (group dispatch heap, slot-token backpressure) | samples_generator.py:195-241; ppo_trainer_async.py:100-232 | Ray + DeepSpeed weight on 1 GPU; no version tags; per-param RPC sync |

### Recommended single-GPU backend for the exit gate
**TRL `GRPOTrainer` (colocate) driven by a Crucible `rollout_func`.** The async part is Crucible's, inside the rollout function.

**Why:**
1. **It is the only option that runs on one 32 GB card without architectural workarounds.**
   - TRL documents colocate as the 1-GPU mode (docs/source/openenv.md:331-337).
   - OpenRLHF async is impossible on one GPU (§B.7). Its sync colocate is untested there and drags in Ray placement and DeepSpeed.
   - rLLM needs verl (the same weight class as OpenRLHF) or a remote Tinker service.
   - TRL AsyncGRPO and server mode need two GPUs: NCCL on a shared device (docs/source/grpo_trainer.md:262; async_grpo_trainer.md:65).
2. **Weight sync costs nothing:** in-process `load_weights` per parameter plus a prefix-cache reset (vllm_generation.py:509-517). The trainer → inference swap is not a variable in the gate.
3. **The pieces the gate measures already exist:**
   - rollout logprobs with `processed_logprobs` (vllm_generation.py:361);
   - `sampling_logp_difference` = `rollout_probs_diff` (grpo_trainer.py:2789-2800);
   - selectable IS modes (grpo_config.py:920-957);
   - FP32 LM head (`cast_lm_head_to_fp32`);
   - DAPO default loss, `mask_truncated_completions`, `frac_reward_zero_std`.
4. **`rollout_func` keeps Crucible in charge of the episode.** It returns exact token ids, logprobs and an `env_mask` (grpo_trainer.py:2134-2200), so the `crucible.rollout/v1` record is the source of truth and the TITO round-trip test (ARCHITECTURE M12 exit gate) is a direct comparison.

**Making it "colocated async"** (the gate wording; DESIGN §10.3 point 2):
- **Concurrency:** inside `rollout_func`, run the Worlds concurrently on CPU (asyncio; environments off the GPU). Each turn submits token ids to the colocated engine.
  - Simplest: batch every pending turn across live episodes into one `vllm_generation.generate` call per tick, which is TRL's own pattern.
  - Better: a small adapter on `llm.llm_engine` that adds requests and steps the engine, so finished turns return as soon as they finish (inference: needs a vLLM-version-specific shim).
- **Version stamps:** use `trainer.state.global_step` at submit time as `policy_version` on each `Turn`.
- **Carry-over with a cap:** episodes not finished when the generation batch quota is met are **carried over** into the next `rollout_func` call (partial rollout across a weight update; the engine is idle during training, so there is no mixed-version request). Their oldest span version then lags, and `staleness.py` drops them when `v_now − v₀ > τ` (τ = 4 per ARCHITECTURE).
- **Correction:** TIS against the recorded behaviour logprobs is what makes carried-over spans usable. `vllm_importance_sampling_correction=True` uses exactly those logprobs.
- **Gate evidence:** this gives a real staleness histogram and a non-trivial `rollout_probs_diff` on one GPU, satisfying "staleness and `rollout_probs_diff` monitors recorded".
- **Scale-up:** the same Crucible layer later moves unchanged to TRL `AsyncGRPOTrainer` + Harbor/OpenEnv loop-owning mode on 2+ GPUs (docs/source/openenv.md:607-700), or to verl via rLLM-style backends.

**Model and memory:**
- Qwen3-0.6B full fine-tune with fp32 Adam ≈ 9.6 GB + activations, and vLLM at 0.30. This fits with headroom (§C.7).
- Qwen3-1.7B needs LoRA, or bf16 weights with 8-bit Adam, or sleep mode.
- Start with 0.6B for the gate; 1.7B + LoRA is the stretch.

---

## H. Pitfalls (from the code)
1. **Mixed-version completions under `pause(mode="keep")`** (OpenRLHF train_ppo_ray.py:299-301; TRL AsyncGRPO merged sync). The recorded version understates the mix, so the IS ratio is computed against logprobs from two policies. Record the swap point or use abort/resubmit.
2. **Re-tokenization drift between turns:**
   - TRL AsyncGRPO and rLLM re-render the whole history every turn (async_rollout_worker.py:876-941; rllm verl_engine.py:104-105) and silently split rows on drift.
   - Qwen3 templates strip `<think>` from history and end turns with `<|im_end|>\n`. TRL swaps in a prefix-preserving training template only when tools are present (grpo_trainer.py:752-756) and EOS-aligns tool suffixes (:1874-1882).
   - Crucible should keep masking on `prefix_break`, log the rate, and fail the gate if it exceeds a threshold.
3. **Missing token records are easy to produce:** rLLM `OpenAIEngine` without a tokenizer returns empty ids (openai_engine.py:132-144); a vLLM server without `--logprobs-mode processed_logprobs` gives raw logprobs that ignore temperature. Both must mask (rollout.py `valid_for_training`), never default.
4. **Logprob semantics must match:** TRL sets `logprobs_mode="processed_logprobs"` "so temperature scaling … affects the TIS log probs" (vllm_generation.py:361), and OpenRLHF divides trainer logits by temperature (models/actor.py:268-271). If either side is wrong, `rollout_probs_diff` looks like mismatch when it is a units error.
5. **Prefix cache after a sync:** OpenRLHF ignores the reset's return value (ppo_actor.py:410-415). checkpoint-engine never resets. Stale KV under new weights is silent corruption.
6. **dtype skew:** TRL warns when the trainer sends weights in a dtype other than what the server serves (async_grpo/weight_transfer.py:82-91). TRL loads string models in **fp32** by default (trl/trainer/utils.py:1326-1338), which also drives the 5090 memory budget.
7. **NCCL on a shared GPU:** server/async modes put the trainer and vLLM on the same device and fail (docs/source/grpo_trainer.md:262). On one GPU use colocate. vLLM has an `ipc` weight-transfer backend for colocated servers (used by SkyRL: infra-novasky-ai-skyrl/skyrl/backends/skyrl_train/weight_sync/__init__.py:47-48; weight_receivers.py:296), but TRL does not use it at this pin.
8. **`max_completion_length` in TRL multi-turn counts every turn and every tool token** (docs/source/openenv.md:203-214). Long Crucible episodes truncate silently unless it is raised. `tools/loop_exhausted_frac` (AsyncGRPO) or `max_tool_calling_iterations` truncation is "a silent truncation" (async_rollout_worker.py:918-925).
9. **Batch-synchronous tool loops stall on the slowest tool** (trl grpo_trainer.py:1899-1953: sync tools run sequentially per rollout). Use async tools or `rollout_func` with concurrent Worlds.
10. **OpenEnv server concurrency defaults to 1 session.** Training fails unless `max_concurrent_envs ≥ generation_batch_size` (docs/source/openenv.md:572-596). The same applies to a Crucible World server.
11. **Dead or ineffective config:**
    - rLLM `filter_token_mismatch` (base.yaml:109), `agent.trajectory_timeout`, `agent.overlong_filter` are never read;
    - rLLM `RolloutEngine.wake_up/sleep` are no-ops (rollout_engine.py:108-112);
    - OpenRLHF `empty_cache` argument is unused on the worker (vllm_worker_wrap.py:32, 52);
    - checkpoint-engine has no pause or reset.
    Don't infer behaviour from flag names.
12. **The checkpoint-engine in-place pin deletes `/dev/shm` safetensors** (ps.py:318; pin_memory.py:257). Never point it at the only copy of a checkpoint.
13. **OpenRLHF flag drift:** old flat flags are rejected (config.py:12-26). Most blog and tutorial commands are stale. Example agent files have signature bugs (agent_func_openai_server_executor.py:285; agent_func_gem_multiturn.py:49).
14. **Per-turn credit is absent everywhere:**
    - OpenRLHF sums per-turn rewards to one scalar (agent.py:122);
    - TRL broadcasts one advantage per completion;
    - rLLM asserts broadcast (advantage.py:188).
    M6 credit needs rLLM's `use_precomputed_advantage` hook or a TRL `_compute_loss` override. Plan for this in `credit.py`, not as a config flag.
15. **Group size 1** (DESIGN §10.3 point 5, single-feedback enterprise cases) breaks every GRPO trainer here. TRL requires full groups (grpo_config.py:1072-1080), and OpenRLHF group estimators assert `n_samples_per_prompt > 1` (train_ppo_ray.py:656-659). The critic mode needs OpenRLHF/verl GAE, not TRL.
