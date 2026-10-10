# AReaL and ROLL: asynchronous agentic RL infrastructure, read from the code

- **Question:** how do AReaL (inclusionAI; Ant Ling/Ring) and ROLL (Alibaba) place, parallelize, desynchronize, version, correct and recover RL training for long multi-turn agent rollouts? What should Crucible M12 (`train/`: `rollout.py`, `topology.py`, `staleness.py`, `mismatch.py`, `objectives.py`, `credit.py`, `compaction.py`, `monitors.py`, `bridges/`) depend on, port or copy as a pattern?
- **Read:** 2026-10-09, from the local checkouts pinned in `00-source-pins.txt`. Every claim cites file:line. Where a claim is about something **absent**, the grep used is stated.
- **Context read first:** DESIGN §10.3–10.4; ARCHITECTURE "M12 `train`"; `docs/research/2026-10-09-agentic-rl-infra-survey.md`; `src/crucible/train/rollout.py`. Also read: the already-built `src/crucible/train/mismatch.py`, so the recommendations add to it instead of duplicating it.

## Root abbreviations

| Abbrev. | Path | Pin |
|---|---|---|
| `AR` | `RL-Environments/06-frameworks-and-env-code/infra-inclusionai-areal` | 298412a, 2026-10-09 |
| `RO` | `RL-Environments/06-frameworks-and-env-code/infra-alibaba-roll` | 581046a, 2026-09-29 |
| `RR` | `RO/roll` (the Python package) | |

## Licenses

Both are **Apache-2.0** (`AR/LICENSE:1-4`, `RO/LICENSE:1-3`). PORT is allowed with attribution, following the existing convention: `src/crucible/_vendor/<name>/ORIGIN.md`, changed lines tagged `# [crucible]`, and a row in `THIRD_PARTY.md`.

## Short file names used below

**AReaL:**

| Short name | Full path |
|---|---|
| `cli_args.py` | `AR/areal/api/cli_args.py` |
| `workflow_executor.py` | `AR/areal/infra/workflow_executor.py` |
| `staleness_manager.py` | `AR/areal/infra/staleness_manager.py` |
| `async_task_runner.py` | `AR/areal/infra/async_task_runner.py` |
| `remote_inf_engine.py` | `AR/areal/infra/remote_inf_engine.py` |
| `rollout_controller.py` | `AR/areal/infra/controller/rollout_controller.py` |
| `train_controller.py` | `AR/areal/infra/controller/train_controller.py` |
| `rl_trainer.py` | `AR/areal/trainer/rl_trainer.py` |
| `actor.py` | `AR/areal/trainer/ppo/actor.py` |
| `gae.py` | `AR/areal/trainer/ppo/gae.py` |
| `stats.py` | `AR/areal/trainer/ppo/stats.py` |
| `functional.py` | `AR/areal/utils/functional/functional.py` |
| `data.py` | `AR/areal/utils/data.py` |
| `fsdp_engine.py` | `AR/areal/engine/fsdp_engine.py` |
| `megatron_engine.py` | `AR/areal/engine/megatron_engine.py` |
| `sglang_remote.py` | `AR/areal/engine/sglang_remote.py` |
| `client.py`, `cache.py`, `types.py` | `AR/areal/experimental/openai/` |
| `proxy/workflow.py` | `AR/areal/experimental/openai/proxy/workflow.py` |
| `recover.py` | `AR/areal/utils/recover.py` |

**ROLL:**

| Short name | Full path |
|---|---|
| `agentic_pipeline.py` | `RR/pipeline/agentic/agentic_pipeline.py` |
| `agentic_config.py` | `RR/pipeline/agentic/agentic_config.py` |
| `agentic/utils.py` | `RR/pipeline/agentic/utils.py` |
| `rollout_scheduler.py` | `RR/distributed/scheduler/rollout_scheduler.py` |
| `generate_scheduler.py` | `RR/distributed/scheduler/generate_scheduler.py` |
| `router.py` | `RR/distributed/scheduler/router.py` |
| `user_defined_rollout_loop.py` | `RR/distributed/scheduler/user_defined_rollout_loop.py` |
| `resource_manager.py` | `RR/distributed/scheduler/resource_manager.py` |
| `base_config.py` | `RR/configs/base_config.py` |
| `worker_config.py` | `RR/configs/worker_config.py` |
| `base_worker.py` | `RR/pipeline/base_worker.py` |
| `base_pipeline.py` | `RR/pipeline/base_pipeline.py` |
| `rlvr_pipeline.py`, `rlvr_config.py` | `RR/pipeline/rlvr/` |
| `traj_env_manager.py`, `step_env_manager.py`, `agent_native_env_manager.py`, `proxy_env_manager.py`, `message_tracker.py` | `RR/pipeline/agentic/env_manager/` |
| `rock_agent_runner.py`, `push_runner.py` | `RR/pipeline/agentic/agent_runner/rock/` |
| `agentic_actor_pg_worker.py` | `RR/pipeline/agentic/agentic_actor_pg_worker.py` |
| `train_infer_corrections.py` | `RR/utils/train_infer_corrections.py` |
| `functionals.py` | `RR/utils/functionals.py` |
| `dynamic_batching.py` | `RR/utils/dynamic_batching.py` |
| `vllm_strategy.py`, `megatron_strategy.py`, `fsdp2_strategy.py` | `RR/distributed/strategy/` |
| `model_update.py` | `RR/third_party/megatron/model_update.py` |

---

## 0. Findings that change M12's design

1. **Only AReaL records a policy version per generated token, and its versions can change *inside one model call*.** Interruptible generation stamps each resumed segment with the version that produced it (`remote_inf_engine.py:1117-1167`), and dumps carry a run-length `version_rle` (`workflow_executor.py:951-965`). Crucible's `Turn.policy_version` is one value per call (`src/crucible/train/rollout.py:38`), which cannot express this. M12 needs `Turn.version_rle: list[[version, n_tokens]]`, with `oldest_version()` taken over it (§4 R1).
   - ROLL records **no** version at token, turn or segment level. A grep for `policy_version|model_version|weight_version|param_version` under `RR` finds nothing. Staleness is known only per prompt or group start step (`rollout_scheduler.py:213`, `:283`).
2. **"Async" means different things.**
   - **AReaL:** the trainer never waits for generation. In-flight requests are aborted at each weight update and resubmitted with prompt + partial output under the new weights: re-prefill, no KV kept (`remote_inf_engine.py:1105-1191`; `AR/blog/AReaL_v0_3.md:91-99`).
   - **ROLL:** generation runs up to `async_generation_ratio` steps ahead of training, but every weight sync **pauses, aborts and drains all generation** (`rollout_scheduler.py:637-640`). Single-turn RLVR resumes the partial output (`user_defined_rollout_loop.py:57-124`). Agentic **discards the partial turn and regenerates it** (`traj_env_manager.py:116-127`).
3. **Neither runs generation and training at the same time on one shared GPU.**
   - AReaL's colocated mode time-slices: rollout is paused, offloaded, trained, then onloaded (`rl_trainer.py:923-960`, `:745-817`). Over-provisioned results still carry across steps, so η > 0 is meaningful.
   - ROLL forbids agentic async with full colocation outright (`agentic_pipeline.py:895-899`).
   - For Crucible's lab RTX 5090 MVP ("colocated + async partial rollout", DESIGN §10.3 point 2), AReaL's time-sliced pattern with carry-over is the realistic shape. Real overlap needs ≥ 2 GPUs (ROLL "partial GPU mode" or AReaL separation).
4. **Staleness is admission control in both, and neither drops at consumption.**
   - AReaL: `capacity = min(max_concurrent − running, (η + v + 1)·B − (accepted + running))` (`staleness_manager.py:79-113`). It bounds the *count*; a straggler is never cancelled.
   - ROLL: groups whose start step is more than `ratio` steps old are expired and their late results ignored (`rollout_scheduler.py:280-288`, `:537-541`).
   - Crucible's planned GLM rule (drop when w′ − w₀ > τ using the *oldest* span version) is stricter than both and needs finding 1.
5. **Both already implement the mismatch-correction menu Crucible wants; neither enables it by default.**
   - AReaL: `RejectionSamplingConfig` gives TIS, IcePop, KPop and sequence/geometric filters on π_prox/π_behave (`functional.py:185-437`); the production MoE config uses IcePop [0.5, 5] plus an FP32 LM head (`AR/examples/swe/arena_flash_v3.yaml:77-82`, `:121`).
   - ROLL: `train_infer_corrections.py` adds a **per-turn (segment) granularity** that Crucible's `mismatch.py` lacks (`train_infer_corrections.py:14-50`).
6. **Failure semantics conflict with Crucible's "masked, never guessed" rule in both:**
   - AReaL: reward-function timeout → 0.0 (`AR/areal/api/reward_api.py:153-161`); a missing reward → 0.0 (`types.py:244`); context overflow → reward 0 (`proxy/workflow.py:598-617`).
   - ROLL: runner error → score 0 (`rock_agent_runner.py:420-428`); empty episode → score-0 placeholder (`proxy_env_manager.py:402-410`); truncation is trained as a normal sample (`agentic/utils.py:238-263`).
   - AReaL's three-way failure classifier (`proxy/workflow.py:284-323`) is the right *shape* for Crucible's `failure_kind`/`blame`.
7. **Skip-observation GAE exists in both.** AReaL's is O(T) vectorized over the batch (`gae.py:10-41`) and adds turn-level GAE plus truncation bootstrap (`gae.py:27`, `:87-172`). ROLL's is an O(B·T²) Python loop (`functionals.py:724-768`). AReaL's is the one to port for Crucible's critic mode.
8. **ROLL's `MessageTracker`** (`message_tracker.py:145-525`) is the best open implementation of token-in-token-out (TITO) reconstruction for an *opaque* OpenAI-speaking harness:
   - hash-chained message prefixes;
   - verbatim engine tokens and logprobs on assistant turns;
   - delta tokenization of new observations;
   - fork detection when the harness rewrites its history.

   It maps one-to-one onto Crucible's capture proxy, linear delta storage and compaction segments (DESIGN §10.4 points 3–4).

---

## 1. AReaL (`AR`)

**What it is.** A fully asynchronous RL system: rollout workers never wait for the trainer. In-flight generations are interrupted on every weight update and resumed under the new weights, and a decoupled PPO objective absorbs the resulting off-policy data. Two runtimes share one core:
- **SPMD mode** (legacy `local`/`ray`/`slurm` launchers, `torchrun` trainers that each own a `RemoteInfEngine` client);
- **single-controller mode** (`RolloutController` + `TrainController` driving RPC workers, `AR/areal/infra/controller/`).

Both use the same `WorkflowExecutor` → `BatchTaskDispatcher` → `StalenessManager` → `AsyncTaskRunner` stack (`AR/areal/infra/workflow_executor.py`, `staleness_manager.py`, `async_task_runner.py`). Published numbers (v0.3 blog, `AR/blog/AReaL_v0_3.md:175-187`): 1.5B on 16 nodes, 1,000 PPO steps in 14.8 h async vs 41.0 h sync (AIME24 42.2 vs 42.0); 7B on 24 nodes, 25.4 h vs 57.7 h. Interruptible generation alone gave +12% / +17% generation throughput (`:198-201`).

### 1.1 Topology and placement

- **Placement is declared per engine** with a backend string `<backend>:<dims>` (`AR/docs/en/reference/alloc_mode.md:20-60`), parsed to `ModelAllocation` (`AR/areal/api/alloc_mode.py`). `rollout.backend: sglang:d4t2` is 4 servers × TP2; `actor.backend: fsdp:d8` is 8 FSDP ranks. World size = dp × tp × pp × cp; EP folds into the mesh (`alloc_mode.md:73-80`). MoE hybrid `megatron:(attn:d4p2t2c2|ffn:d2p2t4e2)` = MoE Parallel Folding (`alloc_mode.md:133-170`).
- **Disaggregated is the default.** `SchedulingStrategy.type` defaults to `"separation"` (`AR/areal/api/cli_args.py:1253-1257`). Colocation is opt-in: `type: colocation, target: <role>`, with `fork: true` (default) spawning a separate process on the same GPUs (`cli_args.py:1258-1269`). `SchedulerConfig.deploy_mode` also defaults to `"separation"` (`cli_args.py:3465`).
- **Colocated actor+rollout is effectively time-sliced.** The trainer detects it (`AR/areal/trainer/rl_trainer.py:638-643`) and wraps each step in `_onload_rollout` → collect → `_offload_rollout` (`rl_trainer.py:923-960`), where offload = `rollout.pause()` + `pause_generation` + `rollout.offload()` (memory release via SGLang `/release_memory_occupation`, `AR/areal/engine/sglang_remote.py:405-419`). Generation is stopped while training runs; asynchrony there only means leftover over-provisioned results carry to the next batch. The AWEX colocate path (`weight_update_mode=awex`, Megatron + SGLang only, `cli_args.py:1457-1463`) does the GPU handover with tagged offload of `kv_cache`, `weights`, `cuda_graph` (`rl_trainer.py:961-972`) and CUDA-IPC weight transfer (§1.5).
- **Schedulers:** `local` (subprocesses, round-robin GPU assignment, health polling every 1 s, `AR/areal/infra/scheduler/local.py:104-181`), `ray` (placement groups + `NodeAffinitySchedulingStrategy`, `AR/areal/infra/scheduler/ray.py:20,662`; colocated roles are *forked* from the target worker's RPC guard, `ray.py:998-1141`), `slurm` (`AR/areal/infra/scheduler/slurm.py`, apptainer containers, `SchedulingSpec.container_type`, `cli_args.py:1324-1330`). Resources per GPU: `cpu: 8`, `mem: 32` GB by default (`cli_args.py:1273-1284`); the rollout controller multiplies them by the instance size TP×PP (`AR/areal/infra/controller/rollout_controller.py:300-321`).
- **Environment / agent execution is on CPU, inside the rollout worker.** A workflow's `arun_episode` runs as an asyncio task in the `AsyncTaskRunner`'s uvloop thread (`async_task_runner.py:281-300`). Agents run in one of three modes (`AR/areal/experimental/openai/proxy/workflow.py:110-125`): `inline` (same event loop), `subproc` (a `ProcessPoolExecutor`, default 4 workers, env `OPENAI_BASE_URL`/`ANTHROPIC_BASE_URL` pointed at the proxy, `workflow.py:190-203`), or `online` (an external user session). Reward functions run in a shared `ProcessPoolExecutor` sized `(cpu/devices)/2` with a 15 s timeout (`AR/areal/api/reward_api.py:76-96`). Remote sandboxes: a Daytona runner (`AR/areal/infra/sandbox/runner.py:102-133`). There is no separate env-worker pool or env-manager abstraction; the env is user code inside the workflow.
- **Data plane (single-controller):** trajectories stay on the worker that produced them as `RTensor` shards and are fetched lazily by consumers; rejected trajectories' shards are cleared best-effort (`workflow_executor.py:1266-1297`). Batches are dispatched to DP ranks by token-balanced greedy partition, groups never split (`AR/areal/infra/controller/train_controller.py:80-129`).

### 1.2 Training parallelism granularity

- **Engines:** FSDP2 (`AR/areal/engine/fsdp_engine.py`, dims `d,t,c`; `c` = Ulysses sequence parallel, `fsdp_engine.py:92-98`), Megatron (`AR/areal/engine/megatron_engine.py`, dims `d,t,p,c,e`), Archon (experimental, PyTorch-native, `AR/areal/experimental/engine/archon_engine.py`).
- **Batch hierarchy:**
  - **global batch** = `train_dataset.batch_size` prompts (`consumer_batch_size` is wired to it, e.g. `AR/examples/swe/arena_grpo.yaml:22`) × `gconfig.n_samples`. One prompt group is one task and one staleness unit (§1.4).
  - **mini-batches** = `actor.ppo_n_minibatches` optimizer steps per batch, default 4 (`cli_args.py:1804-1810`), split with synchronized counts across DP (`AR/areal/utils/data.py:1050-1080`, called at `AR/areal/trainer/ppo/actor.py:819-823`).
  - **micro-batches** = `mb_spec`: `max_tokens_per_mb` (token budget; `None` → effectively unbounded `int(1e12)`, `data.py:815`, `:901-904`), `n_mbs` (minimum count, default 1), `n_mbs_divisor`, `granularity` (adjacent sequences kept together) and `packing_algorithm` `ffd` (default) or `kk` Karmarkar-Karp (`cli_args.py:108-150`). Allocation: `allocate_balanced_mbs` (`data.py:533-557`), and every DP rank is forced to the same micro-batch count by re-allocating with the max (`data.py:560-574`).
  - **Packing:** sequences are unpadded and packed (`pack_tensor_dict`, `data.py:577`; `cu_seqlens` flows into the loss, `AR/areal/utils/functional/functional.py:463`). `pad_to_maximum` trades speed for fragmentation (`cli_args.py:1410-1418`).
  - **Loss normalization is global-token:** each micro-batch's loss is scaled by `local_weight / total_loss_weight`, where the total is all-reduced across the DP group (`fsdp_engine.py:846-884`, `:2249-2313`). Under rejection-sampling masking the denominator stays the *pre-mask* token count, so masking does not inflate surviving tokens' gradients (`functional.py:502-512`).
- **Tree training** packs sequences that share prefixes into a trie and computes shared prefixes once (FlexAttention block masks on FSDP; mbridge on Megatron). The tau2 example reports up to 10× fewer FLOPs and 7× speedup (`AR/docs/en/reference/tree_training.md:1-45`; `enable_tree_training`, `cli_args.py:1514`).
- **FP32 master weights by default** (`optimizer_dtype="float32"`, grads reduced in fp32, `cli_args.py:1431-1453`).

### 1.3 Rollout parallelism granularity

- **Engines:** SGLang (default) or vLLM, launched as HTTP servers. `d` = number of independent servers, each `t×p` GPUs; SGLang-internal DP-attention/EP do not change AReaL's allocation (`alloc_mode.md:105-118`).
- **One sample per request:** `n_samples != 1` is rejected at the engine (`AR/areal/infra/remote_inf_engine.py:1061-1066`). Groups are formed by running the workflow `n_samples` times concurrently in `GroupedRolloutWorkflow` (`remote_inf_engine.py:75-160`).
- **Routing:** round-robin only (`choose_server`, `remote_inf_engine.py:1021-1038`). A request id (`rid`) is pinned to its server for KV reuse, with a 128-entry LRU (`RID_CACHE_SIZE`, `:69`, `:1090-1100`). Built-in workflows mint a fresh `rid` per call (`AR/areal/workflow/multi_turn.py:80`, `AR/areal/experimental/openai/client.py:1266`), so in SPMD mode the turns of one episode can land on different servers. In single-controller mode a whole task (all `n_samples` of a group) goes to one rollout worker (`rollout_controller.py:1064-1067`), so prefix reuse is per group/episode. The v2 inference service has a separate router with session pinning; its `least_busy` strategy is rejected as "not yet implemented" (`AR/areal/v2/inference_service/router/strategies.py:34-44`, `router/app.py:330-380`).
- **Request-level concurrency** is bounded by `rollout.max_concurrent_rollouts` (default = `consumer_batch_size`, `cli_args.py:2844-2850`), divided by DP world size in SPMD mode (`workflow_executor.py:1203-1210`). Continuous batching and the radix/prefix cache are the inference engine's own; AReaL adds no batching layer. `sglang.max_running_requests`, `mem_fraction_static` and `context_length` pass through (`AR/examples/swe/arena_grpo.yaml:119-127`).
- **Production scale** (SWE, `AR/examples/swe/arena_flash_v3.yaml:29-30`): `sglang:d16t4p1` with `max_concurrent_rollouts: 1500`.

### 1.4 Async design in full

**Pipeline.** The trainer calls `prepare_batch(dataloader, workflow)` (`remote_inf_engine.py:1529-1589`). It hands off to `WorkflowExecutor.prepare_batch`, which keeps a persistent infinite task generator over the dataloader (`workflow_executor.py:1660-1674`) and calls `BatchTaskDispatcher.active_submit_and_wait` (`:751-848`). The dispatcher owns two threads:
- a **producer** `_commit_loop` that pops pending inputs while `staleness_manager.get_capacity() > 0` and the runner is not paused (`:438-472`, `:515-528`);
- a **consumer** `_fetch_loop` that drains finished results from the runner's output queue into `_pending_results` (`:474-513`).

The `AsyncTaskRunner` runs every episode as an asyncio task on a uvloop thread (`async_task_runner.py:310-410`).

**Staleness control (admission, not drop).** `StalenessManager.get_capacity` (`staleness_manager.py:79-113`):

```
capacity = min( max_concurrent_rollouts − running,
                (max_head_offpolicyness + current_version + 1) · consumer_batch_size − (accepted + running) )
```

- A new rollout is admitted only while the total ever accepted or running stays below what the trainer will consume by version `v + η`. A rollout *started* at version v is therefore consumed by a policy at most η versions newer: η bounds the **head** of the trajectory (its first token), hence the name `max_head_offpolicyness`.
- **Rejected rollouts free their slot without counting as accepted** (`on_rollout_rejected`, `staleness_manager.py:159-166`). Dynamic filtering therefore does not eat into the staleness budget.
- **The bound is on the count of in-flight + accepted work, not a hard per-sample bound.** A straggler admitted at version v keeps generating across updates (that is the point of partial rollout) and is consumed whenever it finishes. Nothing cancels it when the version passes v + η, so its head staleness is bounded only by `request_timeout`. The `sample_staleness_*` max metric is where this shows up.
- **Nothing is ever dropped for staleness at consumption time.** There is no consumption-side version check; staleness is only *logged* per token (`_log_version_staleness_stats`, `AR/areal/trainer/ppo/actor.py:1832-1880`: `sample_staleness_*`, `stale_token_fraction`).
- Default η = 0, i.e. synchronous (`cli_args.py:2859-2866`); the docs recommend 2–8 (`AR/docs/en/algorithms/async.md:33-42`). The SWE production config uses 1 (`arena_flash_v3.yaml:64`).
- After checkpoint recovery `accepted` is reset to `version × batch` so the formula does not burst (`staleness_manager.py:115-131`; called at `rl_trainer.py:610-619`).

**Over-provisioning and overflow.**
- `active_submit_and_wait` keeps submitting up to `pending_limit = (η+1)·B` minus pending inputs, also capped by queue room (`workflow_executor.py:791-819`).
- It waits for `batch_size − accepted` results. **Surplus results stay in `_pending_results` for the next batch**, taken oldest-first by creation time and shuffled within the batch (`_select_results`, `:316-334`; put-back at `:676-689`).
- The runner's input/output queues default to `max_concurrent_rollouts × 16` (`:1219`). A full input queue requeues at the front and waits (`:455-466`). A **full output queue is fatal** (`TaskQueueFullError`, `async_task_runner.py:384-399`).
- `dynamic_bs` stops at `accepted + rejected ≥ batch_size` and returns a variable-size batch (`workflow_executor.py:820-845`; `PPOConfig.dynamic_bs`, `cli_args.py:4068-4075`).
- The trainer loops `prepare_batch` until every DP consumer has at least one group. It aborts only if ≥ 8 consecutive empty rounds **and** ≥ 1,800 s have passed, so a post-update all-reject burst does not kill the run (`rl_trainer.py:92-157`; constants `workflow_executor.py:304-305`).

**Interruptible generation and per-token versions** (`RemoteInfEngine.agenerate`, `remote_inf_engine.py:1040-1222`):
1. The trainer finishes `ppo_update`, calls `rollout.pause()` (the dispatcher stops admitting and the runner stops harvesting results, `rl_trainer.py:1182`; `async_task_runner.py:327-329`), then `actor.update_weights(meta)`.
2. Inside `update_weights`, rank 0 calls `rollout_engine.pause_generation()` (`fsdp_engine.py:1695-1698`; Megatron `megatron_engine.py:2819-2820`). On SGLang that is **two** `/pause_generation` calls: the default mode aborts in-flight requests and returns their partial output; then `mode=in_place` raises the scheduler flag that colocated memory release needs (`sglang_remote.py:371-391`). On vLLM, `pause_generation(wait_for_inflight_requests=False, clear_cache=True)` aborts and **clears the prefix cache** (`AR/areal/engine/vllm_ext/areal_vllm_server.py:309-321`).
3. Each aborted request returns `stop_reason="abort"` with the tokens and logprobs it produced so far. The client loop appends them to the accumulated output, **extends the prompt with them** (`req.input_ids += output_tokens`), lowers `max_new_tokens`, sleeps while `workflow_executor.is_paused()`, and resubmits (`remote_inf_engine.py:1105-1191`). The old KV is discarded, so resubmission **re-prefills prompt + partial output under the new weights**; this is not KV-preserving resume.
4. **Per-token version:** each resubmission pins `request_version = self.get_version()` *before* the call and stamps every token of that segment with it (`accumulated_versions.extend([request_version] * len(output_tokens))`, `:1117-1167`). The pin is correct because the trainer publishes the new version (`set_version` on actor, critic, rollout, `rl_trainer.py:684-695`) *before* `rollout.resume()` (`rl_trainer.py:1293-1296`), and the client cannot resubmit while paused.
5. Final `abort` is rewritten to `length` (`:1193-1198`). `ModelResponse.output_versions` travels with the tokens. Workflows write `versions = [-1]*prompt + output_versions` next to `loss_mask` and `logprobs` (`multi_turn.py:107-111`; `AR/areal/experimental/openai/types.py:200-242`).
6. Dumps record `head_version`, `tail_version` and a run-length `version_rle` per sample (`workflow_executor.py:951-965`, `:1120-1142`).

**Decoupled PPO objective** (`AR/areal/trainer/ppo/actor.py:1036-1210`, `AR/areal/utils/functional/functional.py:452-597`):
- Three policies: π_behave (per-token logprobs from the inference engine, `data["logprobs"]`), π_prox (the policy at the start of this training step, recomputed by a forward pass, `rl_trainer.py:1067-1084`), π_θ (being optimized).
- PPO ratio and clip are centred on π_prox: `ratio = exp(logp − prox_logp)`, clipped to `[1−ε, 1+ε_high]`, with optional dual clip `c_clip` (`functional.py:544-563`). The off-policy correction is a separate, detached weight `behave_imp_weight = π_prox / π_behave`, multiplied into the loss (`:565-568`) and computed by `apply_rejection_sampling` (`:185-437`).
- `use_decoupled_loss=False` with `recompute_logprob=True` overwrites behave logprobs with the recomputed ones (standard PPO). Without recompute it trusts the engine's logprobs as π_old (`actor.py:457-470`; `should_compute_prox_logp`, `cli_args.py:2041-2052`).
- **Avoiding the extra forward pass** (`prox_logp_method`, `cli_args.py:1984-1996`):
  - `loglinear`: interpolate per token with α = (v_prox − v_behave)/(v_θ − v_behave) (`actor.py:1371-1452`; docs: 27% faster, GSM8K eval 0.799 vs 0.795, `AR/docs/en/algorithms/prox_approx.md:24-35`, `:76-90`);
  - `reuse_train_logp`: the training forward's detached logprobs serve as π_prox, valid only with `ppo_n_minibatches=1` (`actor.py:1129-1130`), so the weight becomes π_θ(start)/π_behave;
  - `metrics`: recompute and also log the approximation error.
- The production MoE SWE config uses `use_decoupled_loss: true`, `recompute_logprob: false`, `prox_logp_method: reuse_train_logp`, `ppo_n_minibatches: 1` (`arena_flash_v3.yaml:75-76`, `:132-133`).
- Ablation (`AR/blog/AReaL_v0_3.md:224-232`): at η=4, naive PPO collapses to 23.3 AIME24 while decoupled gets 42.2 (sync 42.0). At η=16, 35.8 vs 38.7.

**Group sampling and dynamic filtering in async mode.**
- `GroupedRolloutWorkflow` launches `n_samples` copies of the workflow as sibling asyncio tasks inside one dispatcher task (`remote_inf_engine.py:127-145`). A group therefore finishes when its slowest member finishes (group tail), and one group is one staleness unit.
- Sample failures (`None`) are tolerated down to `min_usable_group_size`. It defaults to 2 when reward/adv normalization uses group statistics, else 1 (`cli_args.py:1912-1918`, `:2027-2040`). `drop_incomplete_group` drops the whole group on any failure (`cli_args.py:270-279`; `remote_inf_engine.py:202-232`).
- Per-group slot-yield stats: `usable_slot_count`, `fully_masked_group`, `pre_filter_*_yield` (`remote_inf_engine.py:95-105`).
- Dynamic filtering is `should_accept_fn(traj)` applied to the whole group after it finishes (`workflow_executor.py:1407-1416`). Examples: tau2 keeps groups with mean reward ≤ 0.95 (`AR/examples/tau2/train.py:62-63`); SWE `filter_mixed_or_penalized_all_wrong` (`AR/examples/swe/filter_function.py:80`). Rejection frees the staleness slot (above), so filtering is staleness-neutral, at the price of wasted generation.

**Buffer semantics.**
- There is no replay buffer with reuse: "data from the replay buffer is used only once" (`AR/blog/AReaL_v0_3.md:105-108`).
- The "buffer" is the dispatcher's `_pending_results` dict plus the runner output queue. Ordering is FIFO by creation time (`_select_results`), so the oldest finished groups are trained first.

### 1.5 Weight synchronization

- **Modes:** `weight_update_mode ∈ {disk, xccl, awex}`, default `xccl` (`cli_args.py:1457-1463`). Bucket size `WeightUpdateMeta.weight_chunked_mem_mb = 1024` MB (`AR/areal/api/io_struct.py:192`).
- **xccl, FSDP** (`fsdp_engine.py:1682-1779`):
  - rank 0 pauses generation, then a barrier;
  - every rank all-gathers each DTensor (`full_tensor()` is collective, `:1422-1436`); only rank 0 casts to bf16 (`_cast_to_compute_dtype`) and fills a bucket;
  - when a bucket would overflow 1 GB, the previous bucket is drained and the new one is broadcast asynchronously on a dedicated CUDA stream ("single-pending-bucket pipelining", `:1739-1756`, `:1442-1528`); the gather of bucket k+1 overlaps the NCCL broadcast of bucket k;
  - the rollout side is told the param specs over HTTP (`update_weights_from_distributed`, `remote_inf_engine.py:1284-1312`) and joins the NCCL group created by `init_weights_update_group`;
  - with inference PP > 1 there is one NCCL group per PP stage, broadcast sequentially (`fsdp_engine.py:1476-1499`);
  - at the end: barrier, `continue_generation`.
  - SGLang requests carry `abort_all_requests: True` (`sglang_remote.py:256-262`).
- **xccl, Megatron** (`megatron_engine.py:2813-2958`): `all_gather_param` across TP/EP → `convert_to_hf` → bucketed broadcast. Expert weights are bucketed separately, budget × EP size (`:2574-2665`). Alternatively it streams via megatron-bridge `export` (`:2915-2958`). PP heads each own an update group matched to inference PP stages (`:2694-2770`).
- **disk:** save an HF checkpoint, publish a name-resolve key, the server loads `/update_weights_from_disk` (`fsdp_engine.py:1781-1810`; `remote_inf_engine.py:1314-1360`); the checkpoint dir is deleted after load (`clear_checkpoint_after_load`, `io_struct.py:203`). This is the only path for SGLang LoRA (`sglang_remote.py:240-252`); LoRA adapters are versioned by name and the last `lora_keep_versions` are kept (`sglang_remote.py:189-225`).
- **awex:**
  - colocated: Megatron params → HF → grouped by shape/dtype → `share_memory_()` → CUDA-IPC handles via a MetaServer. The SGLang-side reader deserializes zero-copy, and an inference-only NCCL group redistributes across inference ranks (`AR/areal/engine/awex/colocate_writer.py:1-20`, `colocate_reader.py:1-30`);
  - separated: optional **sparse delta transfer**. Changed bf16 elements are detected by inverting AdamW from the resident moments, with a dense fallback on unsupported optimizers (`AR/areal/v2/weight_update/awex/delta_detect.py:1-50`), plus a periodic full "anchor" every `weight_update_anchor_interval` deltas (`cli_args.py:1464-1482`).
- **Timing:** weight sync is synchronous at the end of each train step, blocking rollout. Overlap is only within the transfer (bucket pipelining). Checkpoint save happens after the update, except in AWEX colocate where it happens before (`rl_trainer.py:1165-1200`). After an AWEX update `abort_all_requests()` is issued so no request continues with mixed old KV state and new weights (`rl_trainer.py:697-703`).

### 1.6 Agentic / multi-turn workflow API

- **Contract:** `RolloutWorkflow.arun_episode(engine, data) -> dict | None | dict[id, InteractionWithTokenLogpReward]`; `None` means reject (`AR/areal/api/workflow_api.py:14-56`). A tensor trajectory carries `input_ids`, `logprobs`, `loss_mask`, `versions`, `turn_ids`, `rewards`, `attention_mask`, `is_truncated` (`multi_turn.py:132-142`). The docstring defines one `arun_episode` = one logical rollout even when compaction exports several rows, tracked by `RolloutGroup` metadata so reward statistics count each rollout once (`workflow_api.py:24-29`; `AR/areal/utils/data.py:387`).
- **Token-in/token-out:**
  - `ModelRequest.input_ids` → SGLang `/generate` with `input_ids` and `return_logprob: True` (`sglang_remote.py:86-91`); the response's `output_token_logprobs` gives `(logprob, token_id)` pairs (`:144-145`). Servers run with `skip_tokenizer_init: true` in the agentic configs (`arena_grpo.yaml:122`).
  - Multi-turn assembly appends real tokens and asserts the new prompt extends the previous sequence (`multi_turn.py:100-111`).
  - For OpenAI-style agents, `ArealOpenAI` / the proxy keep an `InteractionCache` that builds a parent/child tree by **longest message-prefix match** (`AR/areal/experimental/openai/cache.py:421-510`). In `concat` mode a child's prompt is `parent.input_tokens + parent.output_tokens_without_stop + [eos] + new tokens`, where the new tokens are found by aligning on the k-th EOS in the re-templated full conversation (`client.py:694-768`). This is TITO for the history plus re-tokenization of only the new suffix.
- **Observation masks:** prompt and tool tokens get `loss_mask=0`, `logprob=0.0`, `version=-1`, `turn_id=-1`; generated tokens get `1`, the engine logprob, the version and the turn index (`types.py:200-242`).
  - Defect: when a child's input is not longer than its parent's (a "FIXME: Find out why this happens occasionally"), the parent is masked out and the child trains alone (`types.py:220-242`).
  - Defect: the extra EOS appended after a length-truncated parent is a "TODO: should we mask" (`client.py:735-744`).
- **Rewards:**
  - a reward function via `AsyncRewardWrapper`, where a **timeout returns 0.0** after retries (`reward_api.py:153-161`);
  - an agent's `run()` returns a float (last interaction) or a dict per completion id (`proxy/workflow.py:559-566`);
  - turn credit via `apply_reward_discount`: backward geometric `r_i += γ·r_{i+1}` over insertion order (`cache.py:369-419`), for `individual` export. `concat` exports only leaves (`cache.py:564-670`).
  - `token_rewards` (per-token process rewards) either shape advantages after GAE (default) or are added at turn ends before GAE (`token_rewards_as_adv`, `cli_args.py:1898-1910`; `actor.py:503-542`).
- **Env failures / timeouts:**
  - the proxy workflow classifies agent errors (`proxy/workflow.py:284-323`): `context_overflow` → `model_failure_zero` (trained with reward 0); `system_error` → `system_failure_reject` (dropped); otherwise `unknown_failure_reject`. An agent can override this with `classify_proxy_failure`.
  - Rejection re-raises, the executor turns exceptions into `None` (`workflow_executor.py:1471-1487`), and the group logic drops or keeps the remainder.
  - Per-task wall clock: `rollout.request_timeout` (default 3600 s, `cli_args.py:2918-2920`); on timeout the controller **rejects** (masks) the rollout (`rollout_controller.py:1140-1146`). SWE runs use 7,200–20,000 s (`arena_grpo.yaml:40`; `arena_flash_v3.yaml:66`). HTTP retries default 3 (`cli_args.py:2921-2923`).
- **Truncation:** `is_truncated` drives (a) optional reward masking `mask_no_eos_with_zero` (`cli_args.py:1854-1859`; `actor.py:489-494`) and (b) **value bootstrapping** at the last token (`nextvalues = bootstrap_values * seq_no_eos_mask`, `AR/areal/trainer/ppo/gae.py:27`).
- **Critic mode is real:** token-level GAE **skips non-loss tokens** by carrying `nextvalues`/`lastgaelam` through masked positions (skip-observation GAE, `gae.py:29-38`). Turn-level GAE treats each generated turn as one timestep (`gae_timestep_unit="turn"`, `cli_args.py:1886-1896`; `gae.py:87-172`). `gae_lambda` can be a per-trajectory function of lengths and turn counts (length-adaptive, `cli_args.py:1866-1876`).
- **Black-box agents:** the proxy exposes OpenAI- and Anthropic-compatible endpoints (`subproc` sets both base URLs, `proxy/workflow.py:190-195`). AReaL 2.0's agent service splits Gateway / Router / DataProxy / Worker (`AR/areal/v2/agent_service/README.md:1-60`). The proxy `_grant_capacity` call is how online sessions are admitted inside the staleness window (`proxy/workflow.py:481-495`).

### 1.7 Train–inference mismatch corrections

- **One unified mechanism:** `RejectionSamplingConfig` (`cli_args.py:1633-1700`) on the ratio **π_prox / π_behave**. Note that this ratio conflates staleness drift with engine mismatch (`stats.py:19-22` says so explicitly).
  - `level` token | sequence; `action` mask (reject) | clamp (truncate); `metric` ratio | kl_k1 | kl_k2 | kl_k3 | binary_kl; `agg` sum | mean | max; `upper` and optional `lower`.
  - Sequence-level ratio uses the **geometric mean in log space** to avoid the length trap (`functional.py:267-312`).
  - Log-ratios are computed in fp32 and non-finite values are zeroed (`:229-232`).
- **This one config covers the named methods:**

  | Method | Setting |
  |---|---|
  | TIS | token / clamp / ratio / upper |
  | IcePop | token / mask / ratio / [0.5, 5] — the production MoE config, `arena_flash_v3.yaml:77-82` |
  | KPop | `metric: binary_kl`, bidirectional, `functional.py:248-260` |
  | MIS-style sequence filter | sequence / mask / ratio |

- **M2PO masking** (`m2_threshold`, second-moment trust region, `actor.py:1536-1625`).
- **Loss variants:** CISPO (clipped detached IS weight, every token keeps gradient, recommended ε_low=1.0, ε_high=4.0, `cli_args.py:1944-1955`; `functional.py:678-794`), SAPO (incompatible with decoupled loss, `actor.py:1170-1175`), and GSPO (`importance_sampling_level: sequence`, `functional.py:531-541`).
- **Measurement:** `log_train_inference_stats` logs `logp_diff`, `|diff|`, k1/k2/k3 KL, NLLs and R3-style ratio-tail fractions under `ppo_actor/train_infer` whenever `prox_logp` exists (`AR/areal/trainer/ppo/stats.py:14-60`; called `actor.py:451-456`). Also logged: `behave_imp_weight`, `behave_approx_kl`, `behave_filtered_ratio` (`actor.py:1302-1312`).
- **FP32 LM head:** Megatron `enable_fp32_lm_head` (`cli_args.py:1070-1078`), on in the production MoE config (`arena_flash_v3.yaml:121`). `moe_router_dtype: "fp32"` by default (`cli_args.py:996`).
- **Routing replay: capture only, no replay.** `return_routed_experts` asks SGLang for per-token expert ids (`sglang_remote.py:94-95`, `:116-124`), and they are accumulated across interrupted segments (`remote_inf_engine.py:1150-1172`). No training engine consumes them (grep: no `routed_experts` in `AR/areal/engine/megatron_engine.py`, `fsdp_engine.py`, `trainer/ppo/actor.py`).
- **Log-prob recompute policy:** recompute π_prox every step when `use_decoupled_loss` and the method needs a forward (`cli_args.py:2041-2052`). Behave logprobs are **never** recomputed; they are the engine's sampled values.

### 1.8 Long-tail / load balancing

- **The main long-tail weapon is asynchrony itself:** the trainer consumes the first B finished groups, and long ones keep generating across weight updates (partial rollout). The interruptible-generation ablation shows +12–17% throughput (`AReaL_v0_3.md:198-201`).
- **Over-provisioning:** the pending limit (η+1)·B (`staleness_manager.py:69-77`) plus `max_concurrent_rollouts` (1,500 in SWE).
- **No length-aware scheduling, no per-dataset concurrency caps, no length-bias control.** Short groups finish first and are consumed FIFO by creation time, not finish time (`workflow_executor.py:316-334`). The persistent creation-time ordering partially counteracts short-first bias: an old long group that finishes is taken ahead of newer short ones.
- **Trainer side:** token-balanced DP dispatch (`train_controller.py:80-129`; SPMD `dist_rollout.py:47-97`) and balanced micro-batch packing (FFD/KK, `AR/areal/utils/seqpack.py:196-540`).
- **Routing:** round-robin only (§1.3).

### 1.9 Fault tolerance, checkpointing, recovery

- **Two mechanisms** (`AR/docs/en/reference/checkpointing.md:9-17`): `Saver` (HF export, no optimizer) and `RecoverHandler` (DCP model + optimizer + RNG). `RecoverInfo` holds `last_step_info`, saver/evaluator/stats-logger states, **per-rank dataloader state** and the checkpoint frequency controller (`AR/areal/utils/recover.py:44-98`).
- **Not saved:** in-flight rollouts, pending results, partial trajectories. On restart they are lost and regenerated.
- In single-controller mode the DCP checkpoint is an immutable generation published atomically through a `LATEST` pointer (`checkpointing.md:34-40`; `AR/areal/utils/checkpoint_pointer.py`).
- **Recovery** reloads engines, pushes weights to inference at `recovery_version = global_step + 1`, sets versions, then resumes rollout. AWEX colocate has its own pause/offload order to avoid OOM and deadlock (`recover.py:486-545`).
- **Config:** `RecoverConfig.mode` default `disabled`; `on`/`auto` retries the whole job up to `retries=3` when it ends FAILED, NOT_FOUND or COMPLETED (`cli_args.py:3117-3150`; `AR/areal/infra/launcher/local.py:438-446`). Frequencies come from `freq_epochs` / `freq_steps` / `freq_secs` (`cli_args.py:3055-3077`).
- **Worker-level:** the local scheduler health-checks workers every 1 s (`local.py:1016-1056`). A background-thread exception in the dispatcher fails fast (`workflow_executor.py:393-404`).
- **There is no elastic rollout-worker replacement:** a dead inference server fails requests, which become rejections after `request_retries`.

### 1.10 Key config keys (defaults)

| Key | Default | Location | Controls |
|---|---|---|---|
| `rollout.max_head_offpolicyness` | 0 (sync) | `cli_args.py:2859` | staleness η |
| `rollout.max_concurrent_rollouts` | = `consumer_batch_size` | `:2844` | in-flight groups |
| `rollout.consumer_batch_size` | 1 | `:2855` | B in the capacity formula |
| `rollout.queue_size` | `max_concurrent×16` | `:2851`; `workflow_executor.py:1219` | runner queues |
| `rollout.request_timeout` / `request_retries` | 3600 s / 3 | `:2918-2923` | per-request wall clock |
| `rollout.pause_grace_period` | 0.0 s | `:2924` | sleep after pause |
| `rollout.setup_timeout` | 300 s | `:2906` | server launch |
| `rollout.routing_strategy` | round_robin | `:2996` | v2 router |
| `rollout.return_routed_experts` | False | `:2977` | MoE routing capture |
| `rollout.deterministic_sampling` | False | `:2873` | only valid with η=0 |
| `gconfig.n_samples` | 1 | `:175` | group size |
| `gconfig.drop_incomplete_group` | False | `:270` | — |
| `gconfig.reward_normalization` | False | `:251` | — |
| `actor.ppo_n_minibatches` | 4 | `:1804` | optimizer steps per batch |
| `actor.eps_clip` / `eps_clip_higher` / `c_clip` | 0.2 / None / None | `:1811-1824` | clipping |
| `actor.use_decoupled_loss` / `recompute_logprob` | False / False | `:1956-1967` | decoupled PPO |
| `actor.prox_logp_method` | recompute | `:1984` | π_prox source |
| `actor.rejection_sampling` | None (level token, action mask, metric ratio, agg mean when set) | `:1633`, `:1968` | mismatch correction |
| `actor.importance_sampling_level` | token | `:1976` | GSPO |
| `actor.use_cispo_loss` / `use_sapo_loss` | False | `:1944`, `:1930` | loss variant |
| `actor.m2_threshold` | None | `:1827` | M2PO |
| `actor.mask_no_eos_with_zero` | False | `:1854` | truncation masking |
| `actor.gae_timestep_unit` | token | `:1886` | GAE timestep |
| `actor.min_usable_group_size` | None (→ 2 when group-normed) | `:1912` | — |
| `actor.mb_spec.max_tokens_per_mb` | None | `:123` | token budget |
| `actor.mb_spec.packing_algorithm` | ffd | `:136` | — |
| `actor.weight_update_mode` | xccl | `:1457` | — |
| `WeightUpdateMeta.weight_chunked_mem_mb` | 1024 | `io_struct.py:192` | bucket size |
| `actor.megatron.enable_fp32_lm_head` | False | `:1070` | — |
| `dynamic_bs` | False | `:4068` | — |
| `recover.mode` / `retries` | disabled / 3 | `:3120`, `:3129` | — |
| `scheduling_strategy.type` | separation | `:1253` | placement |

---

## 2. ROLL (`RO`, package `RR`)

**What it is.** A Ray single-controller RL library with two pipelines that matter here:
- **RLVR**: single-turn generate + reward, scheduled by `DynamicSamplingScheduler` with a `ReplayBuffer` (`rlvr_pipeline.py`, `generate_scheduler.py`);
- **Agentic**: multi-turn env managers over GEM-style environments, scheduled by `RolloutScheduler` + `GroupQueueManager` (`agentic_pipeline.py`, `rollout_scheduler.py`).

The asynchrony in this pin is the **ROLL Flash** design (README news line, `RO/README.md:52`). **RollFlow, RollPacker (tail batching) and RollArt are not in this pin:** they appear only as README news lines (`RO/README.md:46`, `:52`, `:54`), and a grep for tail-batching or "rollflow" over `RR` finds nothing. The code and docs give no throughput numbers; the only timing metric is `time/step_model_update` (`agentic_pipeline.py:260`).

### 2.1 Topology and placement

- **Ray single controller.** The driver builds role clusters from `WorkerConfig.device_mapping`: `actor_train`, `actor_infer`, references/teachers, `critic`, an optional `reward` (`agentic_pipeline.py:76-127`).
  - `ResourceManager` makes one placement group per node, with all GPUs and half the CPUs in each bundle (`resource_manager.py:42-49`). GPU workers are placed by global GPU rank (`:112-131`).
  - Singletons are Ray named actors with hard node affinity to the driver node: `RolloutScheduler-train/val` (`agentic_pipeline.py:152-174`), `GroupQueueManager-{mode}` with `max_concurrency = env_num + 1` (`rollout_scheduler.py:584-594`), `RouterManager` (`:596-603`), `RewardScheduler-{name}` (`agentic_pipeline.py:133-149`).
- **Environments are CPU Ray actors** on the GPU nodes' CPUs. A worker with `device_mapping=None` is CPU-only (`worker_config.py:357-365`), and CPU workers are spread round-robin over nodes (`resource_manager.py:132-146`).
  - One `EnvironmentWorker` hosts many env managers, each running `run_rollout_loop` in its own thread (`RR/pipeline/agentic/environment_worker.py:24-32`, `:139-163`). Worker count = `ceil(num_env_groups·(group_size+redundancy) / max_env_num_per_worker)`; the default puts every env in one worker, with a warning (`agentic_config.py:149-166`).
  - Heavy environments run in remote sandboxes (ROCK / Harbor runners, `RR/pipeline/agentic/agent_runner/rock/`).
- **Three GPU layouts**, detected from `device_mapping` (`agentic_pipeline.py:756-900`):
  1. **disjoint** train ∩ infer = ∅: async allowed (`:816-820`);
  2. **partial overlap** train ⊂ infer ("partial GPU mode"): requires infer dp ≥ 2, `async_generation_ratio > 0`, the critic disjoint from train, and ≥ 1 infer rank left active (`:829-886`);
  3. **full colocation**: `assert async_generation_ratio == 0, "colocating mode only support sync/on-policy training"` (`:895-899`).

  RLVR async on shared devices only warns (`rlvr_config.py:221-241`).
- **Partial GPU mode = time-sharing.** Before training, `shrink_sampler` aborts in-flight requests on the infer ranks that sit on training GPUs and offloads them (`agentic_pipeline.py:343-359` → `router.py:322-380`, `:1115-1178`). After the next weight update they are re-expanded, and a proportional share of sticky env→rank mappings is remapped onto them (`agentic_pipeline.py:274-287`; `router.py:382-446`, `:1219-1330`). Both run under a routing lock with a 30 s timeout (`router.py:1106-1113`, `:1210-1217`).
- **Data plane:** batches go to a TransferQueue-backed remote store (`DataProto.to_remote`, `rollout_scheduler.py:707-709`; backend `TransferQueue` with 16 storage units, `base_config.py:95-104`). Validation rollout overlaps training in a thread pool (`agentic_pipeline.py:292-296`, `:319-321`).

### 2.2 Training parallelism granularity

- **Backends:** `megatron_train` (through `RO/mcore_adapter`: TP/PP/VPP/CP/EP/SP via `strategy_config`, DP size validated at `base_config.py:958-966`), `fsdp2_train` (device mesh + Ulysses CP via `ulysses_size`, `fsdp2_strategy.py:54-55`, `:95-104`), `veomni_train`, HF, DeepSpeed (`worker_config.py:21-43`).
- **Batch hierarchy:**
  - global batch = `rollout_batch_size × num_return_sequences` (trajectories, or step samples in step mode);
  - mini-batch = `per_device_train_batch_size × gradient_accumulation_steps × dp`, used in `batch_balance` (`agentic_pipeline.py:528-533`);
  - `ppo_epochs` default 1 (`base_config.py:560`);
  - `adjust_batch` copies, deletes or samples rows so the batch divides the LCM of all batch sizes (mode `batch_adjust_mode`, default `copy`; `agentic_pipeline.py:680-754`, `agentic_config.py:203-205`).
- **DP load balance:** `batch_balance` builds equal-size, workload-balanced partitions with workload = `24576·L + L²`, and places small items at both ends of a partition to shrink PP bubbles (`functionals.py:2189-2229`, ported from verl).
- **Token-budget dynamic batching:** `use_dynamic_batching_in_train/infer`, `max_tokens_per_microbatch_in_*`, `sequence_length_round_in_*` (`worker_config.py:152-196`).
  - Sequences are sorted by length, dealt round-robin to DP shards, then greedily packed under the budget. Under VPP, micro-batches are split until their count is a multiple of PP (`dynamic_batching.py:13-138`).
- **Sequence packing:** `use_sequence_packing` defaults to True (Megatron only); it is mutually exclusive with dynamic batching (`worker_config.py:54-88`, `:206-220`, `:300-322`).
- **Loss scaling:** global-token semantics (`apply_loss_scale` × DP × GA, `worker_config.py:255-267`; Megatron `loss_scale = num_microbatches × dp_world`, `megatron_strategy.py:346`). `loss_agg_mode` defaults to `seq-mean-token-mean` (`base_config.py:622-624`).
- **Memory:** state offload, including a `local_dedup` backend (`base_config.py:224-248`); NCCL comm suspend (`offload_nccl`, `worker_config.py:197-203`); logits chunking (`:230-245`).

### 2.3 Rollout parallelism granularity

- **Engines:** vLLM (patched v0.11/0.12/0.17, `RR/third_party/vllm/`), SGLang, `vllm_omni`, `megatron_infer`, `hf_infer` (`worker_config.py:23-36`). vLLM GPUs per worker = TP × PP (`worker_config.py:330-339`). vLLM DP/EP supported (`vllm_strategy.py:204-212`, `:267-306`). `enable_prefix_caching` defaults to True (`vllm_strategy.py:331`).
- **Request-level continuous batching, token-in:** each env or prompt sends one request through `RouterClient.generate_request` (`router.py:675-705`) as `TokensPrompt(prompt_token_ids=…)` (`vllm_strategy.py:525-543`).
  - Responses carry `output_token_ids` and sampled-token logprobs (`:555-592`; `logprobs=0` returns the sampled token's logprob, `RR/configs/generating_args.py:66-69`) and MoE `routed_experts` (`vllm_strategy.py:575-596`).
- **Routers** (`router.py:79-89`):
  - `PromptAffinityRouter` (RLVR default): sticky uid → dp rank, least-loaded with a cap of `max_running_requests` per rank (`router.py:885-991`; `rlvr_config.py:183-185`);
  - `EnvAffinityRouter` (agentic default): sticky env → dp rank, first assignment to the rank with the fewest envs, no rate limit, docstring "Do not support partial rollout now" (`router.py:993-1067`; `agentic_config.py:297-299`);
  - `SglangRouter`: wraps the cache-aware sglang-router; shrink/expand aborts all workers ("FIXME", `router.py:736-883`).
  - `enable_sample_level_affinity` routes the samples of a group independently while keeping each sample's turns sticky (`base_config.py:52-61`; `generate_scheduler.py:1067-1074`).
- **Concurrency:**
  - `max_running_requests` default 128, raised to the engine max; `actor_infer.max_concurrency` = max(max_running + 1, 1000) (`base_config.py:170-176`, `:433-445`). Agentic sets it to #envs + 1 (`agentic_config.py:362-374`).
  - Per-env-tag step limiter `max_env_step_concurrent` (`traj_env_manager.py:62-67`; `RR/utils/env_action_limiter.py:117-122`).

### 2.4 Async design in full

**The knob:**
- `async_generation_ratio: float = 0`, where 0 means synchronous (`base_config.py:552-557`); `async_pipeline = ratio > 0` (`:982-984`).
- RLVR needs ratio ≥ 1.0 and forces group expansion (`rlvr_config.py:221-245`).
- In agentic the ratio must be an **integer**: `for _ in range(self.async_generation_ratio)` (`rollout_scheduler.py:277`). A float raises TypeError despite the `float` annotation.

**Per-step order** (`agentic_pipeline.py:250-330`; RLVR `rlvr_pipeline.py:544-610`):
1. offload train;
2. **suspend + `abort_all` + `wait_complete`** on the router (`rollout_scheduler.py:637-640`; RLVR suspends only *generate* requests and lets reward requests run, `generate_scheduler.py:554-579`);
3. model update (`agentic_pipeline.py:257-259`);
4. reload infer;
5. `get_batch`, which resumes the router (`rollout_scheduler.py:675-677`).

In async mode, generation keeps running while ref, old-logprob and train run. In sync mode it is suspended and offloaded instead (`agentic_pipeline.py:323-330`). The RouterManager blocks new sends while suspended and tracks in-flight requests (`router.py:154-201`).

**RLVR: `ReplayBuffer`**, transactional per prompt (`generate_scheduler.py:156-442`):
- `advance_step` grows the admission budget by `batch_size` per step, plus `ceil(batch × ratio)` on the first step (`:232-252`);
- `poll` admits a prompt iff running + completed < budget, or into an extra pool of `max_additional_running_prompts` (`:254-277`);
- `begin` stamps `sampling_start_step` (`:279-302`); `commit` / `abort` (`:312-349`);
- `get_batch` reads groups in step order from `current − ceil(ratio)`, taking a fractional share of the oldest (`:351-407`);
- `gc()` aborts and deletes groups with step ≤ current − ceil(ratio), and the scheduler cancels their tasks (`:409-442`, `:568-572`);
- logged as `scheduler/off_policy_ratio` = mean(current − start_step) (`:755-779`).

**RLVR partial rollout.**
- With `collect_unfinished = async_pipeline` (`rlvr_pipeline.py:544`), vLLM uses `RequestOutputKind.CUMULATIVE` (`vllm_strategy.py:721-723`), so an abort returns the tokens generated so far.
- `postprocess_paused_data` appends them to the prompt as `pre_output_token_ids`, `pre_output_logprobs`, `pre_routed_experts` and resends under the new weights, which is a re-prefill (`user_defined_rollout_loop.py:57-124`, `:296-317`). The final output concatenates `pre_*` with the new segment (`:176-222`).
- Only `num_return_sequences=1` per request (`:63-65`).
- **No version per segment:** behaviour logprobs from different policies are concatenated into one `infer_logprobs` vector (`:184-190`).

**Agentic: env-level async + `GroupQueue`.**
- **Env managers run free:** each loops reset → `make_decision` → step until terminated or `MAX_LENGTH`, then `put`s the trajectory, with no cross-env barrier (`traj_env_manager.py:94-143`).
- **`GroupQueue` per env group** (`rollout_scheduler.py:217-364`):
  - `advance_step` creates `max_traj_per_env` episode groups per step, plus `ratio×` on the first step (`:262-292`). In async, `max_traj_per_env = ceil(rollout_batch_size / (num_env_groups·group_size))` (`agentic_config.py:318-326`).
  - **Expiry:** groups with `step − create_step > ratio` are removed, and late rollouts for them are silently ignored (`rollout_scheduler.py:280-288`, `:331-332`).
  - `get_episode_id` hands an env the first group with fewer than `group_size + group_size_redundancy` running (`:294-324`).
  - A group completes at `group_size` arrivals (**first K of K+r**). `GroupFilter` may drop it, in which case a replacement group is created at the current step (`:330-349`).
- **`GroupQueueManager.get_batch`** waits FIRST_COMPLETED across group queues and keeps pending gets across calls. It drops groups older than `ratio` (`:537-541`), truncates each to `group_size` (`:543`) and stops at `batch_size` trajectories (`:493-556`). Overflow groups stay queued until they expire.
- **Abort semantics:**
  - `PolicyProxy.generate` returns None on abort (`RR/pipeline/agentic/llm_proxy/policy_proxy.py:32-35`), and `make_decision` → ABORT skips `step()`. The loop then **regenerates the same turn from scratch** under the new weights; earlier turns keep their old-version tokens and logprobs, untagged (`traj_env_manager.py:116-127`, `:241-242`).
  - `ProxyEnvManager` retries once on abort and then raises (`proxy_env_manager.py:260-271`).
- **Straggler cancellation:** the ROCK runners poll `is_episode_done` every 30 s and cancel the agent job and sandbox once their group is complete or expired (`push_runner.py:10`, `:58-72`; `rock_agent_runner.py:85-132`; `rollout_scheduler.py:461-465`).
- **Hang detection:** `EnvActivityMonitor` *logs* envs inactive for more than `hung_timeout` (3600 s, polled every 30 s) and takes no action (`rollout_scheduler.py:31-207`; `agentic_config.py:177-182`).

**Env managers:**
- `TrajEnvManager` (default): one sample per trajectory, TITO concat `Σ(prompt_ids_t + response_ids_t)` (`traj_env_manager.py:281-288`, `:307-396`; `RR/pipeline/agentic/env_manager/token_mask_utils.py:12-37`).
- `StepEnvManager` (GiGPO): one sample per step; history re-rendered as text; `state_hash` groups (`step_env_manager.py:16-141`).
- `AgentNativeStepEnvManager` (SWE / TerminalBench): message list → `apply_chat_template(…, tools)` per step (`agent_native_env_manager.py:23-431`).
- `ProxyEnvManager` + `MessageTracker` for black-box agents (§2.6).
- **GEM envs:** `gem.make(env_type, …)` (`traj_env_manager.py:72`). Registry: sokoban, frozen_lake, MCP sokoban, math/code/qa, sandbox, deepeyes, ROCK TerminalBench, geo3k, openreward, atropos, webshop (`RR/pipeline/agentic/env/__init__.py:1-50`).

**Reward workers:**
- RLVR: per-domain reward clusters. Each prompt's reward call goes round-robin to a domain worker and overlaps other prompts' generation (`generate_scheduler.py:474-477`, `:1099-1122`; `RR/distributed/scheduler/reward_scheduler.py:16-83`).
- Agentic: the env returns step rewards and `episode_score = Σ` (`traj_env_manager.py:318-319`). An optional GPU judge is exposed as the named actor `RewardScheduler-{name}` with env-affinity routing (`agentic_pipeline.py:114-149`).

### 2.5 Weight synchronization

- `ModelUpdateGroup(src=actor_train, tgt=actor_infer, frequency=model_update_frequency=1)` (`RR/distributed/executor/model_update_group.py:9-43`; `worker_config.py:132`). It runs at the **start of every step, before rollout** (`agentic_pipeline.py:257-259`; `base_pipeline.py:144-149`).
- **Megatron → HF in buckets of `model_update_buffer_size_mb = 1024`** (`base_config.py:266-269`): async all-gathers per TP (and EP for experts) → HF names (`model_update.py:49-130`, `:148-169`). Per-PP-stage metadata is gathered once at setup and each bucket is broadcast from its owning stage (`:198-284`).
- **Colocated:** each train rank serializes its bucket as CUDA IPC handles (`RR/utils/send_recv_utils.py:157-190`). A gloo gather collects them to the infer-TP leader, which loads them (`model_update.py:331-367`, `:441-479`). Ray refs are pipelined one bucket deep (`:462-464`).
- **Separated:** the (dp=0, tp=0) rank of each PP/EP stage NCCL-broadcasts to all infer GPUs (`:369-439`). A **global Ray `Locker` serializes broadcasts across stages** (`:493-497`).
- FSDP2 has the same structure (`RR/third_party/fsdp2/model_update.py:23-440`). LoRA syncs adapters only (`model_update.py:445-509`).
- **No overlap with generation:** generation is paused and drained first (§2.4). The prefix cache is reset on every infer load and offload (`vllm_strategy.py:608-622`).

### 2.6 Agentic / multi-turn API

- **Contract:** `BaseEnvManager.run_rollout_loop / reset / step / make_decision / format_messages / formulate_rollouts / update_step / stop` (`RR/pipeline/agentic/env_manager/base_env_manager.py:23-65`), chosen per env tag (`agentic_config.py:379-427`).
  - Group members share env seeds: `group_seed = base + group random seed`, `episode_seed = group_seed + episode_id` (`traj_env_manager.py:94-109`, `:157`).
  - LLM proxies: `policy`, `openai`, `random` (`agentic_config.py:112-115`).
- **TITO for black-box agents (`MessageTracker`):** the agent speaks OpenAI chat to a `ProxyServer` (`proxy_env_manager.py:163-338`), and the tracker (`message_tracker.py:145-525`) does the following:
  - pydantic-normalizes messages (`:57-107`) and SHA-256 hash-chains them as `(index, role, content, reasoning, tool_calls, pre_hash)` (`:113-139`);
  - tokenizes only *new* messages, with a dummy-prefix-then-slice trick (`:216-317`);
  - records **assistant tokens and logprobs verbatim from the engine** (`record_response`, `:334-372`), canonicalizing tool-call ids so re-sent messages hash the same (`:80-91`);
  - detects forks when history is rewritten (`:277-282`, `:444-473`). In traj mode with `enable_fork=False` **only the longest branch is kept** (`:393-400`);
  - sets `response_mask = 1` only on assistant items that have engine logprobs (`:504-513`);
  - inserts a hard-coded `"\n"` token after each assistant message (`:162`, `:326-327`), which assumes a Qwen-style template.
- **Observation masks:** observations and tool outputs are prompt tokens with `response_mask = 0` in every manager (`traj_env_manager.py:328-329`; `step_env_manager.py:78`; `message_tracker.py:504-508`). The loss uses `response_mask[:, 1:]` (`base_worker.py:238`).
- **Truncation:**
  - `max_steps` → `terminated = truncated = True` (`traj_env_manager.py:200-203`);
  - context overflow → `MAX_LENGTH`, still emitted (`:129`, `:223-226`);
  - `get_agentic_response_level_mask` is a **no-op** (`max_len_mask` is a TODO; every sample keeps mask 1, `agentic/utils.py:238-263`).

  Truncated episodes are trained with their score.
- **Env failures / timeouts:**
  - sandbox reset failure → placeholder (all-zero masks, score 0, `drop_flag`) plus retry backoff of 10 s, then `min(3600, 10·n)` (`agent_native_env_manager.py:48-66`, `:433-488`). `GroupFilter` drops `drop_flag` groups **only while the global drop ratio stays ≤ 50%** (`:492-521`);
  - ROCK runner exception → `status="RunnerError", score=0.0` (`rock_agent_runner.py:413-434`), which trains as 0;
  - empty trajectory → placeholder with score 0 (`proxy_env_manager.py:402-410`);
  - task timeout 1800 s (`rock_agent_runner.py:390`); driver RPC timeout 3600 s (`base_config.py:117-120`).
- **Multi-turn advantages:** `adv_estimator` ∈ {gae, **skip_obs_gae**, reinforce, grpo, gigpo, step_reinforce, agentic_reinforce, …} (`base_config.py:598-600`).
  - Grouping keys: `traj_group_id` / `tags` / batch / `state_group_id` (`agentic_config.py:19-109`; `agentic/utils.py:100-154`).
  - `agentic_reinforce` discounts per contiguous response segment, i.e. per turn (`agentic/utils.py:392-458`).
  - Critic: `critic_warmup`, `critic_epochs` (`base_config.py:614-618`).
- **Group sampling and dynamic filtering in async:**
  - `group_size` envs share config and seed; `group_size_redundancy` over-provisions and the first finishers win (`rollout_scheduler.py:307`, `:336`). `rollout_batch_size % group_size == 0` is enforced (`agentic_config.py:330-337`).
  - Filtering is the user `GroupFilter` hook (no-op default, `agentic_pipeline.py:1090-1101`); a filtered group is replaced by a new one at the current step (`rollout_scheduler.py:340-346`).
  - RLVR: `query_filter` (mean/std per domain) after the whole group; a filtered prompt aborts and frees a budget slot (`user_defined_rollout_loop.py:29-52`, `:254-258`; `generate_scheduler.py:946-964`).

### 2.7 Train–inference mismatch, logprob recompute, routing replay

- **Old-logprob recompute policy** (`base_config.py:752-766`, `:946-980`):
  - forced when backward steps per rank > 1, `init_kl_coef > 0`, step-level estimators, or R2 replay;
  - disabled for `pg_variant="sao"`;
  - `force_disable_old_logprobs_recompute` overrides everything.
  - **Without recompute, `old_log_probs = log_probs.detach()`** of the current forward (`base_worker.py:213-216`). On the first epoch the PPO ratio is therefore identically 1.
- **`train_infer_corrections.py`:**
  - ratio = exp(old − infer) and diff = p_old − p_infer at **token / geometric (sequence mean) / sequence (sum) / segment (per contiguous response run, i.e. per turn)** granularity (`:14-50`);
  - a detached `is_weight` clamped at `upper_bound = 1.2` (`base_config.py:474-491`; `:134-151`);
  - a `filters` list of ratio bands [0.8, 1.2] and/or diff bands [−0.2, 0.2] that zero `response_mask` (`base_config.py:494-529`; `:153-188`);
  - **metrics always on:** ratio and diff means per granularity, k3 KL, ratio min/max, clip fraction, mask pass rate (`:96-132`, `:190-194`).
  - **Both corrections default off.** Example: segment ratio [0.8, 1.05] + diff [−0.1, 0.1] (`RO/examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake-pg_var_is_correct.yaml:67-76`).
- **Objective variants** (`agentic_actor_pg_worker.py:66-102`):
  - `vanilla`, `ppo` (clip 0.2/0.2, dual-clip, asymmetric range, `base_config.py:565-572`, `:625`);
  - `tis` (clamp ratio to [0, 1] × A × logπ, `:222-266`);
  - `topr`; `cispo` (clip 1 ± 0.1, stop-grad, `:359-433`); `kimi15` (`:435-473`);
  - `sao` DIS: ratio π_θ/π_rollout, zero weight outside [0.7, 6.0] (`:68-70`, `:475-499`);
  - `ratio_type="segment"` gives a per-turn geometric ratio (`:71-75`).
- **FP32 LM head:** `logits_in_fp32` (`worker_config.py:223-228`; `megatron_strategy.py:918`).
- **Routing replay** (`worker_config.py:10-18`, `:248-253`, `:341-355`; `megatron_strategy.py:154-218`; `RR/third_party/megatron/router_replay_utils.py`, from verl):
  - **R2** replays the trainer's own old-logprob routing (needs recompute);
  - **R3** replays the inference engine's `routed_experts`, carried through partial-rollout concatenation (`user_defined_rollout_loop.py:76-84`, `:192-199`).
  - Default `disable`.

### 2.8 Long-tail / load balancing

- Async lookahead hides generation tails behind training.
- Over-provisioning: RLVR extra prompts (`generate_scheduler.py:254-263`); agentic first-K-of-(K+r) with loser cancellation (`rollout_scheduler.py:307`; `push_runner.py:58-72`).
- RLVR partial rollout lets long generations survive weight updates.
- Partial-GPU shrink/expand gives training GPUs to generation between train phases (§2.1).
- Inference balance: least-loaded + caps (PromptAffinity), fewest-env sticky (EnvAffinity), cache-aware sglang-router.
- Trainer balance: `batch_balance` + PP bubble ordering (`functionals.py:2189-2229`); token-budget dynamic batching (`dynamic_batching.py:13-138`).
- **No** length-aware or shortest-job-first scheduling, no per-dataset concurrency caps, no tail batching in this pin, and no short-first length-bias control.

### 2.9 Fault tolerance, checkpointing, recovery

- **Checkpoints:** every `save_steps` (50) and at the last step. Each cluster saves through its strategy. Pipeline `WorkerState` (step, log history, RLVR scheduler state, RNG) goes to JSON. Upload is async; `max_ckpt_to_keep` prunes (`base_pipeline.py:151-212`; `base_worker.py:345-366`; `base_config.py:151-158`).
- **Resume:** `resume_from_checkpoint` or `auto_resume` (`base_pipeline.py:69-89`). RLVR restores its dataset iterator and `next_prompt_id` (`generate_scheduler.py:507-525`, `:919-923`).
- **Agentic rollout state is not checkpointed,** and the train seed is re-randomized per run (`rollout_scheduler.py:672`).
- **No Ray actor restarts** (no `max_restarts` under `RR`). A crashed role crashes the pipeline. Local handling only: env reset backoff, runner cancel/cleanup (`rock_agent_runner.py:93-132`), hang logging, `rpc_timeout`.
- **Rollout dump/mock** for precision alignment (`base_config.py:17-30`; `rollout_scheduler.py:651-653`, `:704-705`).

### 2.10 Key config keys (defaults)

| Key | Default | Where |
|---|---|---|
| `async_generation_ratio` | 0 (sync); agentic needs an int | `base_config.py:552-557`; `rollout_scheduler.py:277` |
| `generate_opt_level` | 1 (async requires 1) | `base_config.py:65-70`, `:727` |
| `max_running_requests` | 128 (raised to the engine max) | `base_config.py:170-176`, `:433-440` |
| `is_use_additional_prompts` / `max_additional_running_prompts` | False / 16 | `base_config.py:79-85` |
| `router_args.router_name` | PromptAffinity (RLVR) / EnvAffinity (agentic) | `rlvr_config.py:183-185`; `agentic_config.py:297-299` |
| `router_args.enable_sample_level_affinity` | False | `base_config.py:52-61` |
| `model_update_buffer_size_mb` | 1024 | `base_config.py:266-269` |
| `actor_train.model_update_frequency` | 1 | `worker_config.py:132` |
| `enable_old_logprobs_recompute` / `force_disable_old_logprobs_recompute` | False / False (auto rules) | `base_config.py:629-630`, `:752-766` |
| `train_infer_correction.is_weight` | off; token; ub 1.2; detached | `base_config.py:474-491` |
| `train_infer_correction.filters` | [] (each: ratio [0.8, 1.2], diff [−0.2, 0.2]) | `base_config.py:494-529` |
| `pg_variant` | vanilla | `agentic_actor_pg_worker.py:66` |
| `pg_clip`, `use_pg_clip_range`, `dual_clip_loss` | 0.2, False, False | `base_config.py:565-572`, `:625` |
| `loss_agg_mode` | seq-mean-token-mean | `base_config.py:622-624` |
| `adv_estimator` | gae | `base_config.py:598-600` |
| `ratio_type` | token | `agentic_config.py:209` |
| `critic_warmup` / `critic_epochs` | 0 / 1 | `base_config.py:614-618` |
| `actor_train.router_replay.mode` | disable | `worker_config.py:10-18` |
| `logits_in_fp32` | False | `worker_config.py:223-228` |
| `use_sequence_packing` | True (Megatron) | `worker_config.py:206-213` |
| `use_dynamic_batching_in_*` / `max_tokens_per_microbatch_in_*` | False / unset | `worker_config.py:152-196` |
| `train_env_manager.{num_env_groups, group_size, group_size_redundancy, max_env_num_per_worker, max_traj_per_env}` | 128, 1, 0, 0 (= all), −1 (auto) | `agentic_config.py:118-147` |
| `group_filter_cls` | no-op `GroupFilter` | `agentic_config.py:144-147` |
| `env_monitor.{enable, monitor_interval, hung_timeout}` | True, 30 s, 3600 s | `agentic_config.py:177-182` |
| `batch_adjust_mode` | copy | `agentic_config.py:203-205` |
| `save_steps`, `max_ckpt_to_keep`, `resume_from_checkpoint`, `auto_resume` | 50, 0, False, False | `base_config.py:151-191` |
| `rpc_timeout` | 3600 s | `base_config.py:117-120` |
| proxy `trajectory_mode`, `enable_fork`, `reward_granularity`, `task_timeout_sec` | traj, False, pass_rate, 1800 | `proxy_env_manager.py:91-93`, `:119-123`; `rock_agent_runner.py:390` |

---

## 3. Comparison: AReaL vs ROLL

| Dimension | AReaL @298412a | ROLL @581046a |
|---|---|---|
| License | Apache-2.0 | Apache-2.0 |
| Control plane | SPMD launchers or a single controller (`RolloutController` / `TrainController` + RPC workers); schedulers local / ray / slurm | Ray single controller; named-actor schedulers on the driver node |
| Default placement | Separation (`cli_args.py:1253-1257`) | Derived from `device_mapping`: disjoint, partial overlap or colocated (`agentic_pipeline.py:756-900`) |
| Colocated train+infer | Supported, time-sliced (pause → offload → train → onload), with TMS offload or AWEX CUDA-IPC (`rl_trainer.py:923-972`) | Supported, but **sync only** for agentic (`agentic_pipeline.py:895-899`) |
| Partial-overlap mode | No (separation or colocation) | **Yes**: partial GPU shrink/expand of infer ranks (`router.py:322-446`) |
| Env workers | Inside the rollout worker's asyncio loop (inline), a process pool (`subproc`, 4), or an external session; Daytona sandbox | CPU Ray actors, many env managers × threads per worker; ROCK/Harbor sandboxes |
| Train backends | FSDP2 (d,t,c=Ulysses), Megatron (d,t,p,c,e, MoE folding), Archon | Megatron (mcore_adapter: TP/PP/VPP/CP/EP), FSDP2 (+Ulysses), VeOmni, HF, DeepSpeed |
| Batch hierarchy | prompts × `n_samples` → `ppo_n_minibatches` (4) → token-budget micro-batches | `rollout_batch_size × n` → `per_device × GA × dp` mini-batches → micro-batches; `ppo_epochs` |
| Token-budget batching | `mb_spec.max_tokens_per_mb`, FFD or Karmarkar-Karp, synchronized micro-batch count across DP (`data.py:533-574`) | `use_dynamic_batching_*`, `max_tokens_per_microbatch_*`, VPP-aware (`dynamic_batching.py:13-138`) |
| Packing | Always unpadded/packed (`cu_seqlens`); **tree training** for shared prefixes | `use_sequence_packing` (Megatron, default on), exclusive with dynamic batching |
| DP balance | Token-balanced greedy partition, groups atomic (`train_controller.py:80-129`) | Workload `24576·L + L²` + PP bubble ordering (`functionals.py:2189-2229`) |
| Loss normalization | Global token count across DP; pre-mask denominator | Global-token scaling; `seq-mean-token-mean` default |
| Rollout engines | SGLang (default), vLLM | vLLM (patched), SGLang, vllm_omni, megatron/hf infer |
| Routing | Round-robin; `rid` → server pinning (128 LRU); single controller pins a group to one worker | Prompt-affinity (least-loaded + caps), env-affinity (sticky), sglang-router (cache-aware) |
| Concurrency cap | `max_concurrent_rollouts` (default B) | `max_running_requests` (128+), `max_concurrency`, per-env step limiter |
| Async knob | `max_head_offpolicyness` η (default 0) | `async_generation_ratio` (default 0; agentic int) |
| Staleness rule | Admission: `(η + v + 1)·B − (accepted + running)` (`staleness_manager.py:79-113`); no consumption drop | RLVR: budget + gc of prompts older than `ceil(ratio)` steps; agentic: group expiry at `> ratio` steps since create |
| Staleness unit | Prompt group (task) | Prompt (RLVR) / episode group (agentic) |
| In-flight weight update | **Yes**: abort, then resubmit prompt + partial output under the new weights (re-prefill) | **No**: generation is paused and drained before the sync |
| Partial-rollout continuation | Every engine call, single- and multi-turn | RLVR only; agentic regenerates the aborted turn |
| Version per token | **Yes** (`output_versions`, `versions` tensor, `version_rle` in dumps) | **No** |
| Off-policy objective | Decoupled PPO (π_prox centre, detached π_prox/π_behave weight); loglinear π_prox approximation; `reuse_train_logp` | Standard PPO with optional recompute; without recompute, old = current (ratio 1); SAO DIS uses π_rollout |
| Mismatch corrections | `RejectionSamplingConfig`: token/seq × mask/clamp × ratio/k1/k2/k3/binary_kl → TIS, IcePop, KPop, MIS-like; M2PO; CISPO; SAPO; GSPO | `train_infer_corrections`: token/geometric/sequence/**segment** × IS weight + ratio/diff band filters; tis, cispo, kimi15, topr, sao |
| Default correction | None (η=0, decoupled off) | None (IS weight and filters off) |
| Mismatch metrics | `ppo_actor/train_infer/*` (logp diff, k1/k2/k3, R3 tails), `behave_*` | ratio/diff per granularity, k3, clip fraction, pass rate (always on) |
| FP32 LM head | `megatron.enable_fp32_lm_head`; router fp32 | `logits_in_fp32` |
| Routing replay | Captures SGLang `routed_experts` across segments; **no replay in training** | R2 (trainer) and R3 (engine) replay, carried through partial rollout |
| Buffer | Pending-results dict + runner queues; FIFO by creation time; used once | RLVR `ReplayBuffer` per start step; agentic `GroupQueue` per env group |
| Overflow | Surplus results carry to the next batch; output-queue-full is fatal | Surplus groups carry until they expire, then are silently dropped |
| Group sampling | `GroupedRolloutWorkflow`: n concurrent siblings, `min_usable_group_size`, `drop_incomplete_group` | `group_size` + `group_size_redundancy` (first K of K+r) |
| Dynamic filtering | `should_accept_fn` on the finished group; rejection frees the staleness slot; `dynamic_bs`; stall detector | `GroupFilter` (agentic) / `query_filter` (RLVR); filtered → replacement at the current step |
| Weight sync | xccl NCCL broadcast with 1 GB buckets and a single pending bucket pipelined on a side stream; disk; AWEX CUDA-IPC (colocated) and sparse delta (separated) | Megatron/FSDP2 → HF 1 GB buckets; colocated CUDA-IPC via gloo gather; separated NCCL broadcast serialized by a global lock |
| Sync timing | End of step, blocking rollout (paused) | Start of step, after generation is drained |
| Multi-turn API | `RolloutWorkflow.arun_episode`; OpenAI/Anthropic proxy for any agent; `InteractionCache` tree | `BaseEnvManager` family (traj, step, agent-native, proxy); GEM envs |
| TITO for black-box agents | Proxy + parent-prefix token concat (EOS alignment), `concat` / `individual` export | `MessageTracker` hash chain, verbatim assistant tokens, fork detection |
| Observation masks | `loss_mask=0`, `version=-1`, `turn_id=-1` on prompt/tool tokens | `response_mask=0` on observations |
| Env failure handling | Classifier: overflow → reward 0; system/unknown → reject; timeout → reject; reward-function timeout → 0.0 | Runner error → 0; empty → 0 placeholder; reset failures dropped only up to 50% |
| Truncation | `is_truncated` → optional mask (`mask_no_eos_with_zero`) + **value bootstrap** | Trained as normal (mask hook is a no-op) |
| Critic / skip-obs GAE | Yes: O(T) masked carry; turn-level GAE; length-adaptive λ | Yes: `skip_obs_gae` as an O(B·T²) loop; `agentic_reinforce` per-turn discount |
| Long tail | Async + interruptible + over-provisioning; FIFO by creation | Lookahead + first-K-of-(K+r) + cancellation; partial-GPU expansion |
| Length-aware scheduling | None | None |
| Checkpoint / resume | DCP + dataloader per rank; atomic `LATEST` generation; job retries (3) | Strategy checkpoints + `WorkerState` JSON; RLVR scheduler state; auto-resume |
| Rollout state on resume | Lost (in-flight and pending) | RLVR iterator restored; agentic queue lost, seeds re-randomized |
| Worker fault tolerance | Health polls; fail-fast; no elastic replacement | None (no Ray restarts); hang only logged |
| Published numbers | 2.77× (v0.3); 14.8 h vs 41.0 h (1.5B), 25.4 h vs 57.7 h (7B); +12/17% from interruption; loglinear 27% faster | None in code or docs |

---

## 4. Recommendations for Crucible M12

Crucible's rule (memory "build on frameworks"): every module reuses frameworks by DEPEND, PORT or PATTERN. Both repos are Apache-2.0, so PORT is allowed. Rows are ordered by priority for the M12 MVP exit gate (ARCHITECTURE M12: contract tests plus one colocated async GRPO run on the lab RTX 5090).

### 4.1 Contract changes (do first)

**R1. Version per token run (PATTERN, AReaL).** Add `Turn.version_rle: list[tuple[int | str, int]] | None`, run-length (version, n_completion_tokens), summing to `len(completion_token_ids)`.
- Keep `policy_version` as the version of the *first* token, i.e. the head. `oldest_version()` becomes `min` over all runs; `token_complete()` requires the runs to cover the completion.
- **Source:** AReaL stamps per-segment versions in `remote_inf_engine.py:1117-1167` and compresses them to `version_rle` in `workflow_executor.py:951-965`.
- **Why:** with interruptible generation one model call spans versions, and the GLM w₀ drop rule plus AReaL's loglinear π_prox both need per-token versions.
- **Test:** a fake engine aborts mid-call; the resumed tokens must carry the new version.

**R2. Ordering invariant for version publication (PATTERN, AReaL).** Publish the new version on every rollout client *before* resuming admission and generation, and make clients wait while paused before pinning a version.
- **Source:** `rl_trainer.py:684-695` (`set_version` on actor, critic, rollout) precedes `rollout.resume()` at `:1296`; the client loop sleeps while paused before reading `get_version()` (`remote_inf_engine.py:1112-1121`).
- **Test:** M12 `topology.py` must assert this ordering; it is the only thing that makes R1's stamps correct.

**R3. Failure classification at the bridge (PATTERN, AReaL; anti-pattern, ROLL).** Map Crucible's `failure_kind` / `blame` to three dispositions:
- `model_failure_zero`: agent-caused, scored 0, e.g. `agent_broke_env` or a context overflow the policy caused;
- `system_failure_reject`: infra, masked;
- `unknown_failure_reject`: masked plus an alarm.

**Source:** `proxy/workflow.py:284-323`. **Do not** copy AReaL's reward-function timeout → 0.0 (`AR/areal/api/reward_api.py:153-161`), its missing reward → 0.0 (`types.py:244`), or ROLL's runner error → 0 (`rock_agent_runner.py:420-428`): in Crucible these are `reward=None` → masked (`rollout.py` `valid_for_training`).

### 4.2 `staleness.py`

**R4. Admission gate (PORT, AReaL `StalenessManager`, 182 lines, `staleness_manager.py:20-182`).** Take:
- the capacity formula;
- the rule that rejected rollouts free their slot without counting as accepted (`:159-166`);
- the post-recovery reset (`:115-131`).

Then add what AReaL lacks: a **consumption-time drop**, `drop if current_version − record.oldest_version() > τ` (GLM rule, DESIGN §10.3 point 3), counted in a `dropped_stale` metric. AReaL's bound is on counts only, so a straggler's head staleness is unbounded up to `request_timeout` (§1.4).

**R5. Group expiry with redundancy and straggler cancellation (PATTERN, ROLL `GroupQueue`).**
- **Source:** `rollout_scheduler.py:217-364`, `:493-556`; cancellation in `push_runner.py:58-72`.
- **Use:** GRPO groups of long Crucible episodes. Expire a group when its create version is more than τ behind, and cancel its live Worlds (Crucible's `runtime/checkpoint.py` can snapshot them first).
- **Record** `redundancy = r` and `selection = first_k` on the rollout record and in the length-drift monitor: first-K-of-(K+r) biases toward short episodes (§5, R-P7).

**R6. Batch assembly and stall detection (PATTERN, AReaL).**
- FIFO by creation time with surplus carry-over (`workflow_executor.py:316-334`, `:676-689`).
- The stall rule (≥ 8 empty rounds **and** ≥ 1,800 s) for when dynamic filtering rejects everything (`rl_trainer.py:92-157`).

**R7. Length-bias control (gap in both).** Neither has per-dataset concurrency caps or holds back early short returns (§1.8, §2.8). Crucible implements DESIGN §10.3 point 3 itself. AReaL's creation-time FIFO is the cheapest partial mitigation, so start with it.

### 4.3 `topology.py` (single lab GPU first)

**R8. Colocated, time-sliced async on one GPU (PATTERN, AReaL).** Per step:
1. onload the policy server;
2. collect B groups, admitting up to (τ+1)·B;
3. pause → `pause_generation` → offload;
4. train;
5. weight update;
6. publish version → resume.

- **Source:** `rl_trainer.py:745-817`, `:923-960`, `:1182-1296`.
- **SGLang detail to copy:** the pause is **two** `/pause_generation` calls, the default abort mode then `mode=in_place`, before memory can be released (`sglang_remote.py:371-391`).
- **Effect:** generation does not overlap training on one GPU. Async only buys partial-rollout carry-over. State that in DESIGN §10.3 point 2.
- **Overlap needs a second GPU:** AReaL separation (`sglang:d1` + `fsdp:d1`), or ROLL-style partial GPU mode (§2.1) as the "disaggregated later" pattern.
- Environments stay on CPU (both frameworks agree, and Crucible Worlds already run in CPU sandboxes).

**R9. Interruptible generation in Crucible's policy client / capture proxy (PATTERN, AReaL).**
- **Source:** `remote_inf_engine.py:1105-1198`: on `stop_reason == "abort"`, append the partial tokens and logprobs, extend `input_ids`, lower `max_new_tokens`, wait while paused, resubmit, and rewrite a final abort to `length`.
- **Record one `Turn` per logical model call** with `version_rle` (R1), not one per HTTP request. Otherwise `prefix_break` logic sees spurious calls.
- Use the engine's own pause/abort endpoints (SGLang `/pause_generation`, `/continue_generation`; vLLM `pause_generation(clear_cache=True)`, `AR/areal/engine/vllm_ext/areal_vllm_server.py:309-331`).
- **Expect re-prefill cost** on every update. AReaL does not keep KV across updates, and the vLLM path clears the prefix cache. This interacts with 128k-context Crucible episodes: one re-prefill per in-flight episode per step.

### 4.4 `rollout.py` / capture and TITO

**R10. Port `MessageTracker` for opaque harnesses (PORT, ROLL, `message_tracker.py:57-525`)** into `_vendor/roll_message_tracker/`, with these changes:
- (a) keep **every** branch and map each to a Crucible `Segment`; a branch is a compaction or history rewrite, so do not keep only the longest (`:393-400`);
- (b) replace the hard-coded `"\n"` separator (`:162`, `:326-327`) with one derived from the chat template, checked once at startup against `apply_chat_template`;
- (c) attach `version_rle` (R1) to each recorded assistant item;
- (d) emit `prompt_is_delta` / `prefix_break` from the hash-chain match length.

This is the missing piece behind DESIGN §10.3's "rollouts missing token records are masked, never guessed" for black-box harnesses (Claude Code, Codex, mini-swe-agent via M10).

**R11. Parent-prefix concat for OpenAI-client agents (PATTERN, AReaL).**
- **Source:** `client.py:694-768` (reuse parent tokens verbatim, re-template only the suffix, align on the k-th EOS) and the prefix-tree building in `cache.py:421-510`.
- **Do not** copy the silent fallback that masks the parent when the child's input is not longer (`types.py:220-242`, "FIXME"). In Crucible that case is a `prefix_break` and masks the rollout.

### 4.5 `mismatch.py` (already built; additions only)

**R12. Segment (per-call) granularity for every gate (PATTERN, ROLL).**
- **Source:** `train_infer_corrections.py:14-50`: ratio/diff at token / geometric / sequence / segment, where segment = a contiguous response run, i.e. one model call.
- **Change:** add `granularity ∈ {token, sequence, call}` to `Correction`, using `Turn.call_id` spans. MIS-PO's trajectory gate and HisPO's sequence gate then have a per-call variant for 1,000-call episodes, where one bad call should not mask the whole trajectory.
- **Also take:** ROLL's always-on metric set (`:96-132`, `:190-194`) for `monitors.py`.

**R13. Loglinear π_old approximation (PORT, AReaL `compute_prox_logp_approximations`, `actor.py:1371-1452`).**
- **Change:** add `old_logprob_source="loglinear"`, which needs R1's per-token versions.
- **Why:** it skips the recompute forward pass, which matters on one GPU (the docs claim 27% faster, `AR/docs/en/algorithms/prox_approx.md:29`).
- **Caveat:** `reuse_train_logp` (`actor.py:1129-1130`) is equivalent to Crucible's existing `recompute_current` and is valid only with one optimizer step per batch.

**R14. Keep the three-policy separation explicit (PATTERN, AReaL decoupled PPO, `functional.py:452-597`).** The PPO clip is centred on π_prox. The off-policy weight π_prox/π_behave is detached, multiplied in, and has its own mask or clamp. The denominator stays the **pre-mask** token count (`functional.py:502-512`).
- Crucible's `correct()` already returns `kept` and aggregates with the original mask. Add a unit test pinning the pre-mask denominator.
- **Document** that π_prox/π_behave conflates staleness drift with engine mismatch whenever the behaviour version is older than π_prox (AReaL says so in `stats.py:19-22`). The HisPO r_dis/r_stale split in `mismatch.py` is only clean for tokens whose version equals the recompute version.

**R15. Routing replay (PATTERN, ROLL R3; AReaL capture only).** For MoE policies only: carry `routed_experts` through interrupted segments (AReaL `remote_inf_engine.py:1150-1172`; ROLL `user_defined_rollout_loop.py:76-84`) into `Turn.routing`, and replay as in ROLL `megatron_strategy.py:154-218`. Not needed for the dense MVP policy.

### 4.6 `objectives.py` (critic mode) and `credit.py`

**R16. Skip-observation, turn-level GAE with truncation bootstrap (PORT, AReaL `gae.py:10-172`)** for the single-feedback critic mode (ARCHITECTURE M12 `objectives.py`):
- token GAE carries `nextvalues`/`lastgaelam` through `loss_mask = 0` positions (`:29-38`);
- `gae_timestep_unit="turn"` (`:87-172`) maps directly to Crucible's per-call credit;
- truncated rollouts bootstrap from the last value (`:27`), which implements DESIGN §10.3 point 6, "truncation is not failure", for the critic.

Prefer it over ROLL's O(B·T²) loop (`functionals.py:724-768`). Cross-segment GAE (DESIGN §10.4 "remaining") = run turn-level GAE over the concatenated per-call timeline with a segment-boundary discount. Neither framework has it.

**R17. Per-turn process rewards (PATTERN, AReaL `token_rewards`, `actor.py:503-542`).** Two modes: process rewards shape advantages *after* GAE (default, kept out of returns), or are added at turn ends *before* GAE. These are the two places Crucible's ΔΦ per-call credit can enter (M6 → tokens). Pick the first for GRPO (shaping, zero-mean) and the second for the critic.

**R18. Per-turn discounting (PATTERN, ROLL `agentic_reinforce`, `agentic/utils.py:392-458`)** as a critic-free baseline alongside M6 turn advantages.

### 4.7 Trainer-side parallelism (when M12 leaves one GPU)

**R19. DEPEND on the frameworks' engines, not their RL loops.** Packing, token-budget micro-batching and DP balancing are commodities:
- AReaL: FFD/KK, synchronized micro-batch counts, `data.py:533-574`; token-balanced DP, `train_controller.py:80-129`;
- ROLL: `functionals.py:2189-2229`.

Whichever trainer M12 bridges to (verl, prime-rl, or AReaL below) supplies them. Do not reimplement.

**R20. AReaL as an optional async bridge (DEPEND, `bridges/areal.py`, separate environment like `verifiers`).**
- **What it has:** of the two, only AReaL has the three properties Crucible's contract needs: TITO, a per-token version, and admission-controlled async.
- **Integration surface:** the proxy agent workflow. Crucible's harness is the "agent" whose `base_url` points at the AReaL proxy (`subproc` mode sets `OPENAI_BASE_URL`/`ANTHROPIC_BASE_URL`, `proxy/workflow.py:190-195`). `run()` returns the Crucible reward. A masked Crucible outcome raises a typed exception classified `system_failure_reject` via `classify_proxy_failure` (`:293-317`).
- **Export:** `InteractionWithTokenLogpReward.to_tensor_dict` (`types.py:160-260`) → `crucible.rollout/v1` (`versions` → `version_rle`).
- **Gate:** one GRPO step on the toy pack, with the token-in-token-out round-trip test from the M12 exit gate.
- **Not ROLL for the MVP:** agentic async cannot colocate, there are no versions, and infra failures score 0. Revisit ROLL as a DEPEND only for multi-node GEM-env work, and port its pieces (R5, R10, R12, R18) now.

### 4.8 `monitors.py`

**R21.** Add the following:

| Metric | Source |
|---|---|
| `sample_staleness_{avg,max}` and `stale_token_fraction` | AReaL `actor.py:1832-1880` |
| `train_infer/{logp_diff, kl_k3, ratio tails}` | AReaL `stats.py:14-60` |
| per-granularity ratio/diff and mask pass rate | ROLL `train_infer_corrections.py:96-132` |
| group slot yields (`usable_slot_count`, `fully_masked_group`) | AReaL `remote_inf_engine.py:95-105` |
| `off_policy_ratio` | ROLL `generate_scheduler.py:755-779` |
| `dropped_stale` | new; Crucible R4 |

The pass-rate-jump alarm (DESIGN §10.3 point 8) is in neither framework; it stays Crucible's own.

### 4.9 Fault tolerance

**R22.** Take AReaL's atomic `LATEST`-pointer checkpoint generations and its recovery order (load → push weights at `global_step + 1` → set versions → resume; `recover.py:486-545`; `AR/docs/en/reference/checkpointing.md:34-40`).
- Both frameworks lose in-flight rollouts on restart. Crucible already checkpoints Worlds mid-episode (`runtime/checkpoint.py`, DESIGN §10.4 point 5), so it can do better: persist the pending-results queue as `crucible.rollout/v1` files with their versions, and re-admit them after recovery subject to R4's drop rule.

---

## 5. Pitfalls

### AReaL

- **A-P1. Default η = 0 is synchronous** (`cli_args.py:2859-2866`). Decoupled loss and recompute are also off by default (`:1956-1967`). Async must be configured deliberately.
- **A-P2. The staleness bound is not per sample.** The capacity formula bounds counts. A straggler keeps generating across updates and is never cancelled or dropped (no consumption check under `AR/areal/infra` or `AR/areal/trainer`), so its head staleness is bounded only by `request_timeout` (3600 s default). Watch `version_stats/sample_staleness_*` max.
- **A-P3. Every weight update re-prefills every in-flight sequence.** KV is discarded (`AReaL_v0_3.md:93-95`), and vLLM clears its prefix cache on pause (`areal_vllm_server.py:315-318`). At 128k context and τ ≥ 1 this is large.
- **A-P4. A fresh `rid` per call** (`multi_turn.py:80`; `client.py:1266`) plus round-robin routing in SPMD mode spreads one episode's turns across servers and loses prefix reuse. Single-controller mode pins a whole group to one worker (`rollout_controller.py:1064-1067`).
- **A-P5. Zero-filled rewards:** reward-function timeout → 0.0 after retries; a missing reward → 0.0; context overflow → reward 0 (§0 finding 6). Each one silently injects false negatives into group statistics.
- **A-P6. Concat-mode parent fallback** masks the parent and trains the child alone when token prefixes disagree ("FIXME", `types.py:220-242`). It is logged as a warning, not counted as a failure.
- **A-P7. A full output queue kills the runner** (`async_task_runner.py:384-399`). Size `queue_size` at ≥ (η+1)·B with headroom.
- **A-P8. Routing experts are captured, never replayed.** Enabling `return_routed_experts` costs bandwidth for nothing unless you add replay.
- **A-P9. `rejection_sampling` acts on π_prox/π_behave**, which includes legitimate staleness drift when η > 0. Tight IcePop bounds will mask stale but correct tokens. Its help text says "Only effective when use_decoupled_loss=True" (`cli_args.py:1968-1975`).
- **A-P10. `reuse_train_logp` requires `ppo_n_minibatches=1`** (`cli_args.py:1984-1996`). With more minibatches π_prox is wrong.
- **A-P11. Colocation is time-slicing,** not overlap (§1.1). Recovery loses in-flight work, and the DCP checkpoint is tied to the same parallel layout (`checkpointing.md:37-38`).
- **A-P12. Group = staleness unit = one task.** One slow member holds the whole group's slot and delays its consumption (`remote_inf_engine.py:127-145`).

### ROLL

- **R-P1. No async with full colocation in agentic** (`agentic_pipeline.py:895-899`).
- **R-P2. No per-token or per-turn version.** Partial RLVR continuations concatenate logprobs from different policies (`user_defined_rollout_loop.py:184-190`), and agentic episodes mix turn versions untagged.
- **R-P3. The default async run is uncorrected:** without recompute, old = current, so the ratio is 1 (`base_worker.py:213-216`); IS weight and filters are off (`base_config.py:474-529`).
- **R-P4. Agentic abort discards and regenerates the partial turn** (`traj_env_manager.py:116-127`). ProxyEnvManager retries once, then crashes the episode (`proxy_env_manager.py:260-271`). The prefix cache resets every sync (`vllm_strategy.py:610-622`).
- **R-P5. Infra failures are scored 0** and enter group normalization (`rock_agent_runner.py:420-428`; `proxy_env_manager.py:402-410`; `drop_flag` groups kept above 50%, `agent_native_env_manager.py:492-521`).
- **R-P6. Truncation is trained as failure:** the response-level mask is a no-op (`agentic/utils.py:238-263`).
- **R-P7. First-K-of-(K+r)** (`rollout_scheduler.py:307`, `:336`) prefers fast or short trajectories, with no correction.
- **R-P8. `adjust_batch` "copy" duplicates samples without zeroing their mask** (TODO at `agentic_pipeline.py:741`). "delete" can break group completeness.
- **R-P9. `MessageTracker` caveats:** the Qwen-specific `"\n"`, only the longest branch kept by default, and step-mode prompt re-tokenization (`message_tracker.py:410-442`).
- **R-P10. `async_generation_ratio` is typed `float` but must be an int** in agentic (`rollout_scheduler.py:277`).
- **R-P11. Hung envs are only logged** (`rollout_scheduler.py:154-206`). Without redundancy a hang stalls the group until expiry.
- **R-P12. Weight sync is serialized by a global Ray lock across stages and fully blocks generation** (`model_update.py:493-497`).
- **R-P13. Agentic resume loses queued rollouts and re-randomizes seeds** (`rollout_scheduler.py:672`).
- **R-P14. `skip_obs_gae` is O(B·T²) Python** (`functionals.py:724-768`), unusable at 64k–1M tokens.

### Cross-cutting, for Crucible

- **X-P1. One `Turn` per HTTP request is wrong** under interruptible generation. Record one `Turn` per logical call with `version_rle` (R1, R9).
- **X-P2. Group statistics with masked members.** Both frameworks let partial groups train: AReaL down to `min_usable_group_size` (`cli_args.py:1912-1918`); ROLL placeholders count at score 0. Crucible must recompute group statistics over *valid* members only, and record the usable count (AReaL's slot-yield metrics) next to the zero-variance-group share.
- **X-P3. Dynamic filtering in async wastes generation.** Rejected groups free staleness slots (AReaL) or get replacements (ROLL), but their GPU time is gone. Track filter yield per task (`pre_filter_trainable_slot_yield`, `remote_inf_engine.py:95-105`) and feed it back to the curriculum (learnability, ARCHITECTURE M12 `curriculum.py`) instead of filtering forever.
- **X-P4. Reward services under async.** ROLL keeps reward requests running during weight sync (`generate_scheduler.py:554-561`). Crucible's verifiers (M7 full tier) are slow, so make sure a verifier in flight across an update never re-grades against changed state. Crucible grades the World end state, which is version-independent; keep it that way.
