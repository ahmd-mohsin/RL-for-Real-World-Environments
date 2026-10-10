# M11/M12 research 03: prime-rl, NeMo-RL and SkyRL as RL infrastructure

- **Read:** 2026-10-09.
- **For:** M12 `train` (trainer bridges, topology, staleness, mismatch, objectives) and the M11 `interop` bridges (verifiers, NeMo-Gym, OpenEnv/skyrl-gym exports).
- **Context:** DESIGN §10.3 (the RL infrastructure contract) and §10.4; ARCHITECTURE "M11 `interop`" and "M12 `train`"; `docs/research/2026-10-09-agentic-rl-infra-survey.md`; `src/crucible/train/rollout.py`.

## Source pins

Pins are from `00-source-pins.txt`. Every repo is under `RL-Environments/06-frameworks-and-env-code/`.

| Repo | Commit | License (verified) |
|---|---|---|
| `infra-primeintellect-ai-prime-rl` | f7e8ab6 (2026-10-09) | Apache-2.0 (`LICENSE:1-3`). `parallel_dims.py:1-18` carries a Meta BSD header plus an Apache modification notice. |
| `primeintellect-ai-verifiers` (companion) | ae819de | MIT (`LICENSE:1-3`). Caveat: prime-rl pins its `deps/verifiers` submodule at `da75527`, which is not checked out, so verifiers line numbers may drift slightly from what prime-rl runs. |
| `infra-nvidia-nemo-rl` | c9193bc (2026-10-10) | Apache-2.0 (`LICENSE:1-3`; `pyproject.toml:17`) |
| `nvidia-nemo-gym` (companion) | 3ef478d | Apache-2.0 (`LICENSE:1-3`) |
| `infra-novasky-ai-skyrl` | 3d0ae22 (2026-10-09) | Apache-2.0 at the root (`LICENSE:1-3`). `skyrl-gym/pyproject.toml:13` declares MIT. `skyrl-agent/pyproject.toml:19` points at a `LICENSE` file that does not exist. `skycap` is Apache-2.0. |

## Method and conventions

- Each framework was read in the code by a dedicated pass, citing file:line.
- I then re-checked a sample of the load-bearing claims myself. These are marked ✔ where they matter most:
  - single-GPU feasibility;
  - async guards and asserts;
  - weight-sync selection;
  - the bridge data types;
  - the mismatch losses.
- Paths in each framework section are relative to that repo's root. `cfg:` means `packages/prime-rl-configs/src/prime_rl/configs/` (prime-rl). `vf:` means `../primeintellect-ai-verifiers/`. `gym:` means `../nvidia-nemo-gym/`.
- **[docs]** marks a claim taken only from a README, docs page or code comment. "Not found" means it was searched for and is absent at the pin.

---

## TL;DR (what changes for Crucible)

1. **None of the three frameworks does *time-sliced* colocated async with vLLM on one GPU.**
   - SkyRL asserts `not colocate_all` for both async trainers ✔ (`skyrl/train/fully_async_trainer.py:392`; `examples/train/async/async_trainer.py:24`).
   - NeMo-RL asserts that colocated async requires the **Megatron** generation backend:
     - legacy path ✔ `nemo_rl/algorithms/grpo.py:4794-4795`;
     - SingleController ✔ `nemo_rl/algorithms/single_controller_utils/config.py:1163-1170`.
   - prime-rl has no time-sliced colocation at all: its launcher partitions GPUs into disjoint inference and trainer sets ✔ (`src/prime_rl/utils/process.py:52-60`).
   - **Consequence for DESIGN §10.3 point 2:** on the single lab RTX 5090, "colocated + async" must mean either
     - **space-sliced** (vLLM held at a fixed `gpu_memory_utilization`, with the trainer resident on the same GPU), which works with prime-rl's standalone processes and filesystem weight broadcast; or
     - NeMo-RL with Megatron generation, which is risky on sm_120.

     Time-sliced sleep/wake colocation (SkyRL, NeMo-RL with vLLM) is **synchronous** only.
2. **All three already implement the contract Crucible fixed in §10.3**, with different granularity:
   - token-in/token-out through vLLM's `/inference/v1/generate`;
   - behaviour logprobs carried to the trainer;
   - a per-sample (or per-group) policy-version stamp;
   - `pause_generation(mode="keep")` on weight update, so in-flight requests continue under new weights, which is a partial rollout;
   - a session-sticky router;
   - prefix-cache salting by weight version.

   Crucible's `Turn.policy_version` per call is *finer* than all three:
   - prime-rl has `PolicySpan(start, end)` per episode;
   - SkyRL has `global_step_when_scheduled` per group;
   - NeMo-RL has `trajectory_versions` per group in the legacy path, and per call only in the SingleController token-capture path.

   The GLM w₀ rule ("drop when the oldest version is too old") is exactly prime-rl's `max_off_policy_steps` semantics ✔ (`cfg:orchestrator.py:591-592`).
3. **Logprob policy differs, and it decides which mismatch corrections are meaningful.**
   - prime-rl never recomputes π_old. The ratio is π_θ/μ_rollout, the DIS-style denominator (`trainer/rl/loss.py:31-36`).
   - NeMo-RL always recomputes `prev_logprobs` ✔ (`grpo.py:3680-3682`) and applies TIS/IcePop/seq-mask on prev/gen.
   - SkyRL recomputes by default, but skips the recompute for the rollout-anchored losses `rollout_is`, `dppo`, and `cispo` with anchor=rollout (`trainer.py:1349-1374`). Its fully async mode *requires* one of those (`fully_async_trainer.py:353-409`).
4. **Dense per-token credit (M6 turn advantages) has a native slot in each trainer**, but the plugin story differs:
   - prime-rl's wire has per-token `advantages` ✔ (`src/prime_rl/transports/batch/types.py:70`). `assign_advantages(trace, list[float])` accepts per-sampled-token values (`orchestrator/algo/routing.py:17-39`). The algorithm registry, however, is a closed dict ✔ (`orchestrator/algo/__init__.py:43-52`).
   - SkyRL accepts per-token `rewards` ✔ (`skyrl/train/generators/base.py:43`) and has open registries: `@register_advantage_estimator` ✔ (`ppo_utils.py:518`) and `@register_policy_loss`. Note that GRPO sums token rewards ✔ (`ppo_utils.py:1367`).
   - NeMo-RL's estimator dispatch is a hard-coded if/elif ✔ (`grpo.py:2628-2664`).
5. **For the M12 exit gate on the RTX 5090:**
   - SkyRL runs single-GPU sync colocated GRPO out of the box (`NUM_GPUS=1`). It also ships a 1-GPU multi-turn script.
   - NeMo-RL's `grpo_smoke.yaml` is a ready 1-GPU Qwen3-0.6B smoke test ✔. Its Docker image does not build for sm_120, though ✔ (`docker/Dockerfile:204-205`, `:172`), and its 3rdparty submodules are empty in our checkout ✔.
   - prime-rl explicitly needs 2 GPUs for the `rl` launcher ✔ (README.md:144-148). It can run space-sliced on one GPU with standalone processes; that is untested and not documented.

---

## 1. prime-rl (f7e8ab6)

### 1.1 Process topology

**Processes**

| Process | Role | Launch / transport | Cites |
|---|---|---|---|
| Inference | vLLM behind a Rust `vllm-router` | Router on `server.port` (8000), engine on `backend_port` (8100) | `entrypoints/inference.py:165-239`; `cfg:inference.py:443-450, 530-534` |
| Env servers | One per train/eval source | Separate processes speaking ZMQ ROUTER/DEALER with msgpack. Each writes an OS-assigned port to an address file. | `entrypoints/env_server.py:24-51`; `vf:verifiers/v1/serve/server.py:55-64`; `vf:verifiers/v1/serve/client.py:44-52` |
| Orchestrator | A single asyncio process | Loads tasksets, dispatches episodes, scores, packs and drives weight-version application | `orchestrator/orchestrator.py:1-18` |
| Trainer | FSDP2 training | `torchrun -m prime_rl.trainer.rl.train` | `utils/process.py:63-84`; `entrypoints/rl.py:236-260` |

**Links between them**
- **Admin plane** (`/pause`, `/resume`, `/update_weights`, `/init_broadcaster`, `/load_lora_adapter`, `/liveness`): bypasses the router via `admin_base_url` (`orchestrator/clients.py:105-113, 268-294`; `inference/vllm/server.py:40-118`). The launcher auto-wires it only when `[inference]` is present (`cfg:rl.py:787-802`).
- **Rollouts to trainer:** ZMQ PUB/SUB by default (port 5555, hwm 10, READY barrier on port+1) (`transports/batch/zmq.py:11-45`; `cfg:shared.py:277-286`). The alternative is filesystem `rank_{r}.bin` msgpack per step (`transports/batch/filesystem.py:9-62`).
- **No Ray** (no `import ray`). Multi-node needs SLURM and a shared filesystem (`cfg:inference.py:494-498`; `cfg:shared.py:158`; **[docs]** docs/scaling.md).

**Single-node launcher** (`entrypoints/rl.py:111-267`)
- Partitions visible GPUs into **disjoint** inference GPUs followed by trainer GPUs, and raises if the total exceeds what is visible ✔ (`utils/process.py:52-60`; `rl.py:139-141`).
- With no `[inference]` block, `num_infer_gpus` is treated as 0 ✔ (`rl.py:139`).
- Starts inference, env servers, orchestrator and trainer in that order. Any child failing kills the launch (`utils/process.py:196-204`).
- Defaults: `num_train_gpus=1`, `num_infer_gpus=1` (`cfg:rl.py:105-127`). vLLM DP = `num_infer_gpus/tp` with `api_server_count=dp` (`cfg:rl.py:639-652`).

**"Disaggregated"** in prime-rl means **prefill/decode-disaggregated inference** (NIXL KV transfer, SLURM only) (`cfg:inference.py:391-421, 546-564`). It does not mean trainer/inference separation, which is always the case.

**Multi-node.** `MultiNodeDeploymentConfig` with `num_train_nodes`, `num_infer_nodes`, `num_infer_replicas`, `nodes_per_fsdp_group` and `orchestrator_on_inference` (`cfg:rl.py:130-150, 659-740`). It renders `templates/multi_node_rl.sbatch.j2` (`rl.py:270-488`).

### 1.2 Training parallelism

**Sharding**
- FSDP2 `fully_shard` over DTensor meshes. Mixed precision: bf16 params, fp32 reduce (`trainer/model.py:22, 510-511`).
- `ParallelDims` has `dp_replicate`, `dp_shard`, `cp`, `pp` and `ep`, with **pp hard-coded to 1** (`trainer/parallel_dims.py:36-75, 312-319`).
- **No TP and no PP in the trainer, and no Megatron.**
- EP is `"auto"`: `min(world//dp_replicate, 8)` for MoE (`cfg:trainer.py:306-307`).
- CP uses `ring` (default) or `ulysses` (`cfg:trainer.py:312-316`).

**Memory defaults:** full activation checkpointing, activation offloading, `optim_cpu_offload=True`, fp32 optimization dtype and `torch.compile` on (`cfg:trainer.py:279-321`). `fused_lm_head_token_chunk_size=8192` (`cfg:trainer.py:344`); docs/scaling.md says 1024, which is a discrepancy.

**Batching**
- `orchestrator.batch_size=128` **traces** per step (`cfg:orchestrator.py:570`). The cut happens at `len(pending_batch) >= batch_size`, and surplus carries over (`orchestrator/train_sink.py:155-159, 324-341`).
- Each sample is one flat sequence ≤ `seq_len` (`orchestrator/batch.py:340, 385-388`).
- Samples are **First-Fit-Decreasing packed** into `[1, seq_len]` micro-batches (`batch.py:643-679`).
- Micro-batches are assigned to DP ranks by a **Karmarkar-Karp partition on estimated FLOPs** (linear plus quadratic attention) (`batch.py:133-147, 246-252, 682-694`), with dummy micro-batches for divisibility (`batch.py:814-820`).
- **No mini-batches and no PPO epochs.** Each step accumulates gradients over all micro-batches, then runs one `optimizer.step()` (`trainer/rl/train.py:343-584`).
- Loss components are normalized by global all-reduced denominators (`train.py:302-327, 569-572`).

### 1.3 Rollout parallelism

**Engines.** One vLLM server per node with DP engines behind one router. Defaults: `tp=1`, `dp=1`, `gpu_memory_utilization=0.9`, `enable_prefix_caching=None` (vLLM default) (`cfg:inference.py:86-107`). Unknown `[inference.vllm]` keys pass through (`cfg:inference.py:452-453, 609-623`).

**Load balancing and stickiness**
- Done by the router, not the client. The default policy is `sticky_least_loaded` (`cfg:inference.py:309-318`), launched with `--request-id-headers x-session-id` (`entrypoints/inference.py:165-191`).
- verifiers sends `X-Session-ID` = trace id on every turn ✔ (`vf:verifiers/v1/clients/client.py:16`; `vf:verifiers/v1/clients/train.py:454`).
- Finished sessions call `/finish_session` (`orchestrator/clients.py:86-102`; `dispatcher.py:604-610`).

**Concurrency.** One permit equals one episode.
- The adaptive AIMD cap reacts to engine KV usage: it grows ×1.2 per turnover, soft-trims above 0.8, and hard-trims and cancels above 0.9 (`orchestrator/concurrency.py:1-79`).
- Bounds: `max_inflight=1024`; the initial value is auto (KV capacity / `max_model_len`) (`cfg:orchestrator.py:485-506`).
- Burst cap: `max(group_size, max_inflight//10)` per 5 s window (`dispatcher.py:173-182, 271-278`).
- Optional `tasks_per_minute`.

**Prefix cache.** `cache_salt = str(group.policy_version_at_start)` is applied to live-policy requests ✔ (`dispatcher.py:561-565`; `envs.py:112-116`).
- The salt is fixed per group at dispatch. Turns of an episode that run *after* a weight update therefore still hit KV prefill computed under the old weights. This is my reading of the code, and the same trade-off SkyRL documents as `clear_kv_cache_on_weight_sync=False`.
- The dispatcher fills an open group before opening a new one, which keeps prefix hits (`dispatcher.py:487-508`).

**Continuous batching** is vLLM's own.

### 1.4 Async design

**There is no `async_level` key at this pin.** The controls are:

| Control | Value | What it does | Cite |
|---|---|---|---|
| `TARGET_LAG` | hard-coded 1 ✔ | Batch `step` ships only after inference has applied v{step−1−TARGET_LAG}; the dispatcher is paused when the lead exceeds it | `orchestrator/orchestrator.py:84-87, 593-611, 963-987` |
| `max_off_policy_steps` | 8 ✔ | Trained staleness = v{step−1} − the **oldest** version that generated the rollout, queue time included. Over the bound, the episode is dropped in-flight or in queue. | `cfg:orchestrator.py:591-592`; helpers at `orchestrator/utils.py:42-61` |

**How a step's batch forms**
1. The dispatcher opens a group (task × `group_size`) and stamps `policy_version_at_start`. Each episode is one `env.run` request (`dispatcher.py:510-615`).
2. The sink waits until arrived + failed + cancelled ≥ `group_size` (`train_sink.py:115-153`).
3. The group is scored by the algorithm, then the curriculum gate decides admission.
4. Traces become samples and enter `pending_batch` (`train_sink.py:205-298`).

**Staleness tracking**
- Per episode: `PolicySpan(start, end)` (`dispatcher.py:716-727`; `vf:verifiers/v1/episode.py:31-46`).
- Metrics: `off_policy/{mean,max,in_flight,in_queue}` (`orchestrator.py:677-692`).
- The hard guarantee is a sweep of queued traces (`train_sink.py:161-198`).
- Stale in-flight groups are cancelled just before a weight update (`on_version_pending`, `dispatcher.py:401-439`). A stale cancel voids the group's already-arrived episodes too (`train_sink.py:230-241`).

**On weight update** (`watcher.py:27-109`; `dispatcher.py:415-459`; `clients.py:178-205, 403-421`)
1. The watcher polls the broadcast directory every 0.1 s.
2. It drops stale groups.
3. It stops scheduling.
4. It calls `/pause`, i.e. `pause_generation(mode="keep", clear_cache=False)` ✔ (`inference/vllm/server.py:40-44`), then `/update_weights` and `/resume`.

In-flight requests are kept and continue under new weights: these are **partial rollouts**, and the `PolicySpan` records them. A comment at `clients.py:375-376` says "drains", which conflicts with `mode="keep"`. The keep semantics live in vLLM itself.

**Over-sampling and filtering**
- There is no explicit over-sampling factor.
- `constant_trainer_batch_size=True` drops zero-advantage samples *before* they count toward the batch, so more groups get dispatched (`cfg:orchestrator.py:573-574`; `train_sink.py:29-54, 273-274`).
- Opt-in curriculum (`cfg:orchestrator.py:227-317`):
  - `DifficultyPoolSampler`: easy, normal and hard pools by the latest group-mean reward (`orchestrator/curriculum/samplers/pool.py:18-55`);
  - `AdvRangeGate`: rejects groups with no signal (`orchestrator/curriculum/gates/adv.py:15-36`).

### 1.5 Weight broadcast

**Transport choice** ✔ (`cfg:rl.py:417-440`)
- **NCCL** is the default when the launcher manages inference and LoRA is off.
- **Filesystem** is used with LoRA ("vLLM loads adapters only from a PEFT-shaped directory on disk") or when there is no `[inference]` block.
- NCCL requires ≥2 GPUs single-node ✔ (`cfg:rl.py:288-296`).
- Third option: NIXL. Shardcast: not found.

**Handshake** (`transports/weights/base.py:17-108`)
1. The trainer master resets `broadcasts/step_N`, touches `.sender_ready`, and waits for `.receiver_ready` with a timeout of 1200 s (`cfg:shared.py:41-65`).
2. It touches `.started`, runs the transfer, then touches `.finished`.
3. Broadcast directories N and N−1 are kept.
4. A broadcast happens synchronously after every optimizer step, plus once at startup (`train.py:265-278, 589-605`).

**Filesystem transfer**
- Gathers full weights, converts to HF naming and saves safetensors. LoRA saves only the PEFT adapter (`transports/weights/filesystem.py:21-59`).
- vLLM loads through a worker extension, `FileSystemWeightUpdateWorker.update_weights_from_path` (`inference/vllm/worker/filesystem.py:17-51`).
- Full weights go through pause → update → resume. An adapter goes through `/load_lora_adapter` **without pause** (`filesystem.py:62-76`), reloaded in place under a fixed name (`server.py:60-89`).

**NCCL transfer**
- Uses a `StatelessProcessGroup` + `PyNcclCommunicator` with `world = infer_world + 1` (`transports/weights/nccl.py:126-188`).
- Walks the model layer by layer: DTensor `.full_tensor()` in bf16, HF names, dtype-bucketed broadcast (`nccl.py:41-168`).
- vLLM workers get **full tensors** and call `model.load_weights`, so resharding happens inside vLLM's loaders (`inference/vllm/worker/nccl.py:90-149`).
- The worker extension class is chosen by transport (`server.py:33-37, 203-204`).

**Numbers.** None measured in docs. A code comment says the per-layer gather plus fp8 conversion "peaks ~50 GiB above the resident weights" (`train.py:595-597`). LoRA loads are "single-digit seconds" (`clients.py:437-440`).

### 1.6 Environment integration (verifiers v1)

**Env server**
- Loads the env by id, validates `task_data`, runs `env.run_slot`, and streams trace `delta` frames (`vf:verifiers/v1/serve/server.py:28-123`).
- The orchestrator owns the taskset and ships `task_data` per request (`orchestrator/envs.py:1-18, 90-135`).
- Wire types: `RunRequest{task_data, client, model, sampling}` and `RunResponse{head, traces}` (`vf:verifiers/v1/serve/types.py:10-63`).

**Env groups.** Several `[[orchestrator.train.source]]` entries, each with its own `ratio`, `group_size`, `algo`, `sampling` and `curriculum` (`cfg:orchestrator.py:298-317`), picked by weighted RNG (`orchestrator/train_source.py:38-40`).

**Rewards.** v1 `Task` declares `@reward` and `@metric` hooks. `trace.rewards[name] = Reward(score, weight)`, and `trace.reward` is the weighted sum (`vf:verifiers/v1/task.py:213-284`; `trace.py:135-141, 472-474`).

**TITO.** Black-box harness calls go to a localhost **interception server**, which routes them to `TrainClient` (`vf:verifiers/v1/interception/server.py:1-20`). `TrainClient`:
- renders chat to token ids;
- **bridges** multi-turn prompts onto the previous turn's exact ids ✔ (`renderer.bridge_to_next_turn`, `vf:verifiers/v1/clients/train.py:410`);
- calls `/inference/v1/generate` ✔ (`train.py:323`);
- stores `prompt_ids`, `completion_ids` and `completion_logprobs`.

**Masking and samples**
- `MessageNode` holds `token_ids`, `mask` (True only on sampled tokens), `logprobs` and `advantages` (per-token, same layout as logprobs) ✔ (`vf:verifiers/v1/graph.py:83-143`; `advantages` at :138).
- Each root-to-leaf `Branch` becomes one sample. A prefix break starts a new branch, i.e. a new sample (`vf:verifiers/v1/trace.py:192-266, 515-553`; **[docs]** docs/algorithms.md:493-519).
- Env, tool and user tokens get `mask=False`. A shared sampled node trains only in its first branch (`orchestrator/trajectories.py:105-196`).

**Failures**
- A request that produces no episode is a `DispatchFailure`. It counts toward group completion but is never scored.
- Errored traces and non-ok episodes are **excluded from scoring and from the baseline**, not zeroed (`dispatcher.py:641-666`; `orchestrator/algo/base.py:11-18`; `train_sink.py:243-255`).
- Train timeouts are dropped as errors; eval keeps them as scored (`dispatcher.py:668-674`).

**Timeouts.** The env-server request is **untimed** ✔ ("rollouts run untimed", `vf:verifiers/v1/serve/client.py:101`). Episode and finalize deadlines are opt-in (`vf:verifiers/v1/configs/env.py:22-27`). The router request lifetime is 14400 s (`cfg:inference.py:314`).

### 1.7 Mismatch, logprobs, losses, advantages

**Logprobs.** There is **no recompute of π_old.** One forward pass under v{step−1} is compared with the shipped `inference_logprobs` (`transports/batch/types.py:84-96`; `train.py:442-498`; `loss.py:31-36`). The ratio π_θ/μ therefore covers both numeric mismatch and staleness, which is DIS-style. Per-token temperatures divide the logits (`train.py:461-466`).

**Losses.** The default is **IPO** ✔ (`cfg:trainer.py:626`).

| Loss | Rule | Defaults | Cite |
|---|---|---|---|
| `ipo` | Mask tokens with \|π−μ\| > eps (absolute probability difference, i.e. a binary-TV-style mask); PG = −A·ratio with a capped ratio | eps 0.3 | `cfg:trainer.py:522-531`; `loss.py:145-190` |
| `icepop` | Mask tokens whose ratio is outside the band | [0.2, 5.0] | `cfg:trainer.py:534-550`; `loss.py:193-224` |
| `ppo` | Token-level clipped surrogate | clip [0.8, 1.2] | `loss.py:227-253` |
| `cispo` | Detached clipped IS weight | [0, 5] | `loss.py:256-285` |
| `custom` | Your `fn(LossInputs, **kw) -> LossOutputs` | — | `cfg:trainer.py:588-595`; `loss.py:47-61, 347-356` |

- Fixed side components: `ce` and `ref_kl` (`loss.py:288-344`). `mismatch_kl` is logged only.
- **Not found:** TIS by name, MIS, geometric sequence IS, and any KL-to-reference penalty coefficient.

**System-side mismatch fixes**
- `enable_fp32_lm_head=True` by default ✔ (`cfg:inference.py:475-476`, which injects `hf_overrides head_dtype`).
- `logprobs_mode=processed_logprobs` (`cfg:inference.py:630-632`).
- fp32 MoE router logits.
- Sampling-mask replay for top-p/top-k: training fails if masks are missing (`loss.py:73-91`; `train_sink.py:263-271`).
- MoE router replay (`train.py:356-367`).

**Aggregation.** Global token-mean (`train.py:302-327`), or prompt-mean via `loss_aggregation="prompt"` (`cfg:algorithm.py:207-208`; `orchestrator/algo/grpo.py:56-61`). No sequence-mean.

**Advantages** are computed in the orchestrator.
- Registry: closed dict `ALGORITHM_CLASSES` ✔ (`orchestrator/algo/__init__.py:43-52`) with `grpo` (mean-centred, **no std**, `algo/grpo.py:26-61`), `max_rl`, `rae`, `hierarchical_grpo`, `opd`, `opsd`, `sft`, `echo`, `debug`.
- `assign_advantages(trace, float | list[float])` accepts a **per-sampled-token list** ✔ (`algo/routing.py:17-39`).
- **No critic / PPO-with-value mode.**

### 1.8 Long tail and load balancing

The mechanisms:
- Async overlap: lag 1, staleness 8.
- Batches fill from whichever groups finish first.
- Adaptive concurrency cancels the **youngest** groups first (`concurrency.py`; `dispatcher.py:242-269`).
- Sticky least-loaded routing.
- FLOP-balanced Karmarkar-Karp DP packing.
- Imbalance warnings (`orchestrator.py:698-704`; `train.py:646-652`).

Not found: hedged or duplicate rollouts, or a straggler timeout (env requests are untimed).

### 1.9 Fault tolerance and resume

**Fail-fast.** The launcher exits on any child failure. The orchestrator re-raises if the dispatcher or watcher dies (`orchestrator.py:547-557`). Admin operations retry on 5xx (`clients.py:358-400`). verifiers episode retries default to 0 (`vf:verifiers/v1/configs/retries.py:38-48`). There is no automatic process restart.

**What is checkpointed**
- Trainer: DCP of model, optimizer, scheduler and progress, resumable at any world size (`trainer/ckpt.py:48-54, 183-213`).
- Orchestrator: atomic `progress.pt` plus `train_source.state_dict()` (RNG and curriculum) (`orchestrator/ckpt.py:30-47`).
- **Not checkpointed:** `pending_batch`, partial groups and in-flight episodes.

**Resume** prunes future broadcast and rollout directories and rebroadcasts v{resume_step} (`rl.py:510-530`; `train.py:265-278`; `orchestrator.py:232-247`).

### 1.10 Key config defaults

| Class (cite) | Defaults |
|---|---|
| `RLConfig` (`cfg:rl.py:162`) | `weight_broadcast=None`, which auto-selects NCCL or filesystem; `deployment=SingleNode(1 train, 1 infer)` |
| `OrchestratorConfig` (`cfg:orchestrator.py:518`) | `batch_size=128`, `seq_len=2048`, `max_off_policy_steps=8`, `constant_trainer_batch_size=True`, `max_inflight=1024`, `rollout_transport=zmq` |
| `TrainSourceConfig` (`:298`) | `group_size=1`, `algo=GRPO` |
| Train sampling (`:47`) | `temperature=1.0`, `top_p=1.0`; always requests logprobs |
| `TrainerConfig` (`cfg:trainer.py:619`) | `loss=IPO`, AdamW `lr=1e-6`, `max_norm=1.0` |
| Trainer model (`:269`) | `cp=1`, `ep=auto`, full activation checkpointing plus offload, `optim_cpu_offload=True`, `lora=None` (rank 16, alpha 32 when enabled, `:106`) |
| `VllmConfig` (`cfg:inference.py:44`) | `model="Qwen/Qwen3-0.6B"`, `gpu_memory_utilization=0.9`, `max_loras=1` |
| `InferenceConfig` (`:443`) | `router=vllm-router`, `enable_fp32_lm_head=True` |
| Weight broadcast (`cfg:shared.py:41-78`) | timeout 1200 s, NCCL port 29501 |

### 1.11 One RTX 5090

**Official position**
- README: "the full RL stack ... *requires 2 GPUs*" ✔ (`README.md:144-148`).
- docs/overview.md:25 ✔ says "Single-GPU runs are supported for debugging", but the only single-GPU commands shipped are **SFT** (`examples/basic/reverse-text/README.md:41`; `examples/basic/wordle/README.md:43`).
- "We develop and test on ... RTX 3090/4090/5090" ✔ (`README.md:64`).

**Space-sliced recipe.** Inferred from the code. **Untested and not documented.**
1. `CUDA_VISIBLE_DEVICES=0 uv run inference --vllm.model PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT --vllm.gpu-memory-utilization 0.35 --vllm.max-model-len 4096`.
   - Standalone inference defaults to the filesystem worker extension (`cfg:inference.py:219-220`; `server.py:203-204`).
2. `uv run rl @ examples/basic/reverse-text/rl.toml` with the `[inference]` block removed. Then:
   - `num_infer_gpus=0`, so the trainer takes GPU 0 ✔ (`rl.py:139-140`);
   - broadcast auto-falls back to filesystem ✔ (`cfg:rl.py:425-427`).
3. Because `admin_base_url` is not auto-wired without `[inference]`, set `--orchestrator.model.client.admin-base-url '["http://localhost:8100/v1"]'` (`cfg:rl.py:787-802`; `clients.py:105-113`).

**Memory**
- vLLM has no sleep mode in prime-rl, so the split is static: 0.35 × 32 GB ≈ 11 GB for vLLM.
- The trainer has fp32 master weights and grads, with Adam state on CPU by default. For 1.7B that is about 13.6 GB plus activations (my estimate). 0.6B is comfortable.
- `reverse-text` is the smallest shipped recipe ✔: 0.6B, `seq_len` 2048, 128 completion tokens, batch 128, group 16 (`examples/basic/reverse-text/rl.toml`).
- Wordle uses Qwen3-1.7B on 6+2 GPUs; it needs scaling down.

**What the recipe would give.** Real async on one GPU: lag 1, `max_off_policy_steps`, keep-mode partial rollouts, `PolicySpan` staleness and the `off_policy/*` metrics. That is exactly the M12 exit-gate shape.

**Dependencies.** Python ~3.12, `torch>=2.13` (cu130), `vllm>=0.31.0` (`pyproject.toml:56`), transformers 5.6.2, `vllm-router`. flash-attn 2.8.3 wheels "carry sm_80/90/100/120 cubins" per the pyproject comment. Base image `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` (`Dockerfile.cuda`).

**sm_120 notes**
- `attn="auto"` gives **FA2** on SM12x; FA4 only on SM10x/11x (`trainer/model.py:930-948`). This contradicts the docstring at `cfg:trainer.py:277`.
- FP8 paths need SM100 (`model.py:871-874`). prime-kernels are sm_100a only.
- The quack CuTe RMSNorm is used when capability ≥ 9 and the package is installed (`trainer/models/layers/norms.py:10-21`). Its sm_120 support is unverified; skip the `quack` extra if it fails.

### 1.12 Minimum bridge interface

**Path A: a verifiers v1 `Taskset` + `Env`.**
- `Taskset.load() -> Iterable[Task]`; `Env.run(task, agents)`; rewards via `@reward` or `trace.record_reward` (`vf:verifiers/v1/taskset.py:49-75`; `env.py:79-187`; `task.py:213-284`).
- verifiers' own OpenEnv taskset is the reset/step template (`vf:verifiers/v1/tasksets/openenv/taskset.py:73-134`).
- The consumed output is a `WireEpisode` whose traces carry nodes (`token_ids`, `mask`, `logprobs`) and rewards, with `ok=True` (`vf:verifiers/v1/episode.py:87-110`).
- This is the path DESIGN §10.1 already names (`export verifiers`).

**Path B: feed the trainer directly.** The trainer is algorithm-blind.
- It reads per-rank `MicroBatch` objects ✔ (`src/prime_rl/transports/batch/types.py:84-117`):
  - `input_ids`, `loss_mask`, `advantages`, `inference_logprobs`, `position_ids`, `sequence_lengths`, `temperatures`, `env_names`, `seq_lens`;
  - optional `rl_weights`, `ce_weights`, `ref_kl_weights`, `sampling_mask`, `routed_experts`.
- The easiest producer is `TrainingSample` ✔ (`types.py:31-80`: `token_ids`, `mask`, `logprobs`, `temperatures`, `env_name`, per-token `advantages` at :70), packed with `orchestrator.batch.prepare_batch`.
- Batches arrive over ZMQ, or as filesystem msgpack per step.
- The producer must also consume `broadcasts/step_N` (`.sender_ready` → `.receiver_ready` → `.finished`) and load the HF safetensors into its engine (`base.py:17-26`).
- Caveat: the msgspec structs are `array_like=True` and encoded **positionally**, so the producer must pin the prime-rl version.

---

## 2. NeMo-RL (c9193bc) with NeMo-Gym (3ef478d)

**Two orchestration stacks coexist:**
- **(A) Legacy driver-orchestrated GRPO.** `nemo_rl/algorithms/grpo.py`, sync or `async_grpo`, with Ray `ReplayBuffer` and `AsyncTrajectoryCollector` actors.
- **(B) SingleController (SC).** A CPU-only Ray actor with rollout and train pumps, a TransferQueue data plane, pluggable staleness samplers, fleet health and engine restart. Configured under `async_rl` (`nemo_rl/algorithms/single_controller.py:15-37, 323`).

### 2.1 Process topology

**Controller.** Legacy: the driver process (`grpo.py:566, 870-1240`). SC: `@ray.remote(num_cpus=1, num_gpus=0)`, which carries "control signals and metadata only" (`single_controller.py:15-37, 323`). The TransferQueue backend is `simple` or `mooncake_cpu`, off by default (`data_plane/interfaces.py:131-132`; `examples/configs/grpo_math_1B.yaml:606-609`).

**Ray virtual cluster** (`distributed/virtual_cluster.py:914-1110`)
- Bundles are `{"CPU": max_colocated_worker_groups, "GPU": 1}`, one PACK placement group per node by default, with an optional unified PG.
- The PG timeout is 180 s.
- SLURM via `ray.sub`; K8s under `infra/`.

**Worker groups** (`distributed/worker_groups.py:358-635`)
- One actor per bundle, with `num_gpus = 1/max_colocated_worker_groups`, so colocated policy and vLLM each get 0.5.
- **Every actor runs in its own uv venv** (`distributed/actor_environments.py:43-107`).

**Environments.** Native envs are CPU Ray actors (`environments/utils.py:111-137`). NeMo-Gym runs as a `NemoGym` Ray actor (`max_restarts=0`) that starts Gym's `RunHelper` (`environments/nemo_gym.py:450-457, 653-661`). That in turn launches each Gym server (model proxy, resources, agent) as a `Popen` subprocess with its own venv (`gym:nemo_gym/cli/env.py:469-512`). Gym can be sharded with one actor per node (`environments/nemo_gym_shards.py`).

**Channels**
- Ray RPC and the object store.
- HTTP: Gym ↔ the vLLM OpenAI server.
- NCCL for non-colocated refit (`grpo.py:1851-1866`).
- ZMQ + CUDA IPC handles for colocated refit (`grpo.py:2749-2758`).
- Optional NIXL/UCX, S3/ZMQ sparse delta, TransferQueue/Mooncake.

**Colocated vs not.** The switch is `policy.generation.colocated.enabled`, default true (`grpo_math_1B.yaml:508-516`).
- **Colocated:** `max_colocated_worker_groups=2`; vLLM `enable_sleep_mode`, slept at level 1 (`vllm_worker.py:769, 1581`).
- **Non-colocated:** separate clusters. Single node: inference takes `colocated.resources.gpus_per_node` and training gets the rest (`grpo.py:1026-1071`).

### 2.2 Training parallelism

**Backends.** Two, chosen in `Policy.__init__` (`models/policy/lm_policy.py:157-287`):
- `automodel_cfg` (DTensor/FSDP2 via NeMo-Automodel);
- `megatron_cfg` (Megatron-core via Megatron-Bridge).

**Automodel path**
- TP, CP, EP and HSDP. **PP hard-coded to 1** (`models/automodel/setup.py:498-548`).
- CPU offload raises NotImplementedError at world_size=1 (`:535-540`).

**Megatron path**
- TP, PP, SP, CP (requires packing), MoE ETP/EP, distributed optimizer and virtual PP (`models/megatron/setup.py:1247-1305, 2090-2103, 2517`).

**Batches**
- `local_gbs = gbs // dp`. Per global batch: zero_grad, then forward/backward per micro-batch (accumulate), then clip, then step (`automodel_policy_worker.py:427-566`; `megatron_policy_worker.py:1105-1283`).
- Defaults: 32 prompts × 16 generations = 512 = gbs, i.e. one optimizer step per RL step, with mbs 4 (`grpo_math_1B.yaml:3-4, 147-148`).

**Dynamic batching** (token budgets `train_mb_tokens`, `logprob_mb_tokens`) and **sequence packing** (concatenative, FFD, MFFD, knapsack variants) are mutually exclusive (`data/packing/algorithms.py:25-33`; `lm_policy.py:414-472`). Packing with MFFD is on by default (`grpo_math_1B.yaml:380-386`).

### 2.3 Rollout parallelism

**Backends:** vllm, sglang, megatron, trtllm, dynamo (`grpo.py:1476-1770`).

**vLLM shape**
- `dp = world // (TP·PP)` engines (`models/generation/vllm/vllm_generation.py:184-213`).
- Defaults: TP=PP=EP=1, `gpu_memory_utilization 0.6`, `async_engine: false` (`grpo_math_1B.yaml:481-498`).

**Sync vs async engine**
- Sync uses `vllm.LLM`, with the batch sharded evenly across DP (`vllm_generation.py:970-985`).
- Async uses `AsyncLLM`, one sample per call, round-robin or least-in-flight with fleet health (`vllm_generation.py:1113-1174`; `fleet_health.py:570-611`).
- Per-request timeout: `NRL_VLLM_ASYNC_TIMEOUT_SECONDS=900` (`vllm_generation.py:1197-1262`).

**Prefix caching:** on when compute capability ≥ 8 (`vllm_worker.py:171-175`).

**Sticky multi-turn**
- Native path: **not found.**
- NeMo-Gym path: the vLLM proxy maps `sha256(session_id) % len(clients)` (`gym:responses_api_models/vllm_model/app.py:1701-1709`).

**HTTP for Gym.** `expose_http_server` (requires the async engine; forced on by `setup_nemo_gym_config`, `environments/nemo_gym.py:1515-1542`). It serves `/v1/chat/completions` and `/tokenize` per DP engine (`vllm_worker_async.py:895-1523`).

### 2.4 Async design

**Legacy `AsyncGRPOConfig`** ✔ (`grpo.py:273-287`)

| Key | Default |
|---|---|
| `enabled` | False |
| `max_trajectory_age_steps` | 1 |
| `max_generation_failures` | 0 |
| `in_flight_weight_updates` | False |
| `recompute_kv_cache_after_weight_updates` | False |

**Preconditions**
- An async engine is required.
- **IS correction is mandatory** ✔ (`grpo.py:4724-4726`).
- **Colocated async requires Megatron generation** ✔ (`grpo.py:4794-4795`).
- Dynamic sampling, reward scaling and reward shaping are rejected in async (`examples/run_grpo.py:184-196`).

**Replay buffer** (`algorithms/async_utils/replay_buffer.py:384-673`)
- One entry is one prompt group, with `trajectory_versions` and `target_weight_versions`.
- `sample()` evicts entries older than current − max_age and returns only groups targeted at the current step, FIFO, or None if it is short.
- Size is `prompts × max_age × 2` (`grpo.py:4824-4827`).
- The collector reserves future target steps and gap-fills (`trajectory_collector.py:310-882`).

**Weight update with `in_flight_weight_updates`:** `pause_generation(mode="keep", clear_cache=recompute_kv…)`. Requests are **frozen, not aborted**, and resume afterwards (`vllm_worker_async.py:2284-2310`; `trajectory_collector.py:1048-1153`). Otherwise the collector drains.

**SC samplers** (`algorithms/async_utils/staleness_sampler.py`)
- `in_order` (default) with `max_lookahead_versions: 1` ✔ (`examples/configs/grpo_math_1B_megatron_single_controller.yaml:24-28`).
- `windowed`, with `max_staleness_versions` (over-samples and evicts aged groups, `:308-362`).
- `ready_first`, `weight_fifo`, and a custom `module:Class`.
- SC aborts in-flight native rollouts outside the window (`single_controller.py:4230-4282`); the Gym path skips the abort (TODO).

**Partial rollouts**, meaning continuing a truncated generation later: not found. Keep-mode pause is the mechanism.

**Dynamic sampling (DAPO), sync only:** `use_dynamic_sampling` with `batch_multiplier`, up to 10 generation batches (`grpo.py:1999-2140, 370-378`). `overlong_filtering` zeroes the loss multiplier (`grpo.py:3571-3580`).

### 2.5 Refit (weight sync)

**Abstraction.** `WeightSynchronizer` (`weight_sync/interfaces.py:48-169`), with a factory (`weight_sync/factory.py:15-206`). The default is CUDA IPC over ZMQ when colocated and a packed NCCL broadcast otherwise.

**Colocated sequence** (`weight_sync/ipc_weight_synchronizer.py:20-137`)
1. Optimizer to CPU.
2. vLLM `wake_up(weights)`.
3. Stream through a ping-pong IPC double buffer.
4. Model to CPU.
5. `wake_up(kv_cache)`.

The buffer is `free × 0.3` (`NRL_REFIT_BUFFER_MEMORY_RATIO`).

**Non-colocated NCCL.** Trainer rank 0 is the source. The packed buffer is `min(total × 0.02, 5 GiB)` ✔ (`utils/packed_tensor.py:24-31`), with two alternating buffers.

**Megatron → vLLM.** Megatron-Bridge `export_hf_weights` gathers TP/PP/EP (`megatron_policy_worker.py:3298-3466`). `nccl_reshard` does shard-to-shard transfer with no full gather (experimental).

**Numbers**
- **[comment]** About 1.9 s for 1.5B on GB200 (`single_controller_utils/config.py:336-341`).
- **[docs]** `nccl_reshard`: Qwen3-4B 0.21–0.34 s; DSV3 1.9–2.8 s (`docs/design-docs/nccl-reshard-refit.md:354-362`).

**FP8 generation** asserts IS correction (`grpo.py:1603-1606`). **[docs]** Blackwell does not support FP8 training with FP32 scales (`docs/fp8.md:15-16`).

### 2.6 Environment integration

**Native `EnvironmentInterface`** (`environments/interfaces.py:26-90`)
- `step(message_log_batch, metadata) -> EnvironmentReturn(observations, metadata, next_stop_strings, rewards, terminateds, answers)`, plus `global_post_process_and_metrics`.
- **No `reset`.** The initial state is the dataset `DatumSpec` (`data/interfaces.py:63-71`).

**Native multi-turn loop** (`experience/rollouts.py:928-1176`)
- The assistant message stores the engine's `token_ids` and `generation_logprobs`.
- `env.step` is a blocking `ray.get` **with no timeout** (`:812-843`).
- Rewards are **summed across turns** (`:1043-1061`).
- Observations are tokenized raw, with **no chat template** (TODO) ✔ (`:1066-1076`).
- Overflow truncates the observation and sets `truncated` (`:1078-1093`).

**Loss mask** ✔ (`grpo.py:2222-2254`). 1 only for assistant messages that carry `generation_logprobs`; everything else, including prompt-history assistant turns, gets 0.

**NeMo-Gym path**
- Gym's `vllm_model_for_training.yaml` points at `policy_base_url` with `return_token_id_information: true` (`gym:responses_api_models/vllm_model/configs/vllm_model_for_training.yaml:1-10`).
- Rows are dispatched to Gym's `/run` (seed_session → model ↔ tools loop → `/verify`).
- The **contiguity assert** ✔ requires each turn's `prompt_token_ids` to extend all earlier tokens: "Non-contiguous messages found! ... if reasoning is stripped out" (`environments/nemo_gym.py:1219-1228`).
- NeMo-RL's vLLM server **splices the exact earlier token ids** into re-templated history (`required_prefix_token_ids`; `vllm_worker_async.py:961-973, 1090-1170`).
- `reward` becomes `total_reward`. Optional `mask_sample` and `reward/<name>` components are supported (`rollouts.py:3168-3216`).

**Failures.** Gym 5xx → `GymTransportError` (infra); otherwise `RolloutDataFailure` (`experience/failures.py:84-236`). SC adds `generation_timeout_s`, `env_timeout_s`, `rollout_timeout_s` and `max_row_attempts=3` (`single_controller_utils/config.py:74-103`).

### 2.7 Mismatch, logprobs, losses

**Logprobs.** The trainer **recomputes `prev_logprobs`** by default ✔ (`grpo.py:3680-3682`). Engine logprobs are `generation_logprobs`, the behaviour policy for IS. `force_on_policy_ratio` skips the recompute (`clipped_pg.py:137-140`).

**`ClippedPGLossFn`** (`algorithms/loss/clipped_pg.py:116-179`; `loss_functions.py:171-236`)
- Clip min/max (clip-higher), dual-clip.
- **GSPO** via `sequence_level_importance_ratios`.
- **CISPO** via `use_cispo`.
- REINFORCE/RLOO via `disable_ppo_ratio`.

**IS correction.** `use_importance_sampling_correction` with w = exp(prev − gen) ✔ (`loss_functions.py:720-760`):
- `tis`: clamp;
- `icepop`: zero outside the bounds (reference 0.5–5);
- `seq-mask-tis`: zero sequences by geometric mean (reference 0.999–1.002).

Also `grpo.seq_logprob_error_threshold` (`grpo.py:383-385`), and a vLLM `fp32_lm_head` option ✔ (`lm_policy.py:182-197`).

**Aggregation.** `token_level_loss=True` gives a global token mean; False gives a per-sequence mean.

**Advantage estimators** — hard-coded dispatch ✔ (`grpo.py:2628-2664`): `grpo` (leave-one-out, std), `gdpo`, `reinforce_plus_plus`, `opd`. PPO uses `gae` (`ppo.py`; `models/value`).

**Other objectives:** DPO, RM, SFT, MPO, distillation (top-k, on-policy, cross-tokenizer). Dr.GRPO is not named; it is approximately `normalize_rewards=false` plus token loss.

### 2.8 Long tail and load balancing

- Async lookahead.
- Keep-mode pause.
- The Gym path streams groups as they complete.
- The native path keeps a **full-batch barrier** (`rollouts.py:1720-1735`).
- SC starts streaming training at `min_groups_for_streaming_train`; the windowed sampler aborts stragglers.
- SC failure budgets and a stall watchdog (30 s interval, 600 s timeout) (`single_controller_utils/config.py:132-208, 450-462`).

### 2.9 Fault tolerance and resume

- `CheckpointManager`: `tmp_step_N` → `step_N`, async finalisation, top-k / latest-k retention (`utils/checkpoint.py:151-585`).
- Resume restores `GRPOSaveState` and the dataloader (`grpo.py:425-754`).
- Async: `replay_buffer.pt`, optional and noted as biased toward short rollouts (`grpo.py:193-256`).
- SC: a rollout-recovery ledger and a fleet-health state machine; `EngineSupervisor` restart is off by default (`models/generation/engine_supervisor.py:15-46`).
- **[docs]** `ft_launcher` handles whole-job restarts (`docs/guides/ft-launcher-guide.md`).

### 2.10 Key config defaults

`GRPOConfig` (`grpo.py:330-402`) and `examples/configs/grpo_math_1B.yaml`:

| Area | Defaults |
|---|---|
| Model and batch | Qwen2.5-1.5B; 32 prompts × 16; gbs 512, mbs 4; max seq 512; automodel backend; MFFD packing |
| Generation | vLLM temperature 1.0, `gpu_memory_utilization 0.6`, colocated |
| Loss | KL 0.01 (k3), clip 0.2, IS off, token-level |
| Cluster | `cluster.gpus_per_node: 1` |

### 2.11 One RTX 5090

**Sync colocated runs on one GPU by default.**
- ✔ `uv run examples/run_grpo.py --config examples/configs/grpo_smoke.yaml`: Qwen3-0.6B, GSM8K, 2 prompts × 4, 10 steps, checkpointing off (`examples/configs/grpo_smoke.yaml:1-20`).
- **[docs]** README.md:325-332 says `uv run python examples/run_grpo.py` runs on one GPU.
- A multi-turn variant: `examples/run_grpo_sliding_puzzle.py`.
- A NeMo-Gym 0.6B colocated config exists for distillation (`examples/nemo_gym/distillation_qwen3_0_6b.yaml`: `async_engine: true`, `colocated.enabled: true`). The Gym GRPO example `grpo_workplace_assistant_nemotron_nano_v2_9b.yaml` is colocated with 8 GPUs. A 1-GPU Gym GRPO run needs `policy.model_name=Qwen/Qwen3-0.6B cluster.gpus_per_node=1`; I found no shipped config for it.

**Async on one GPU**
- Only the Megatron generation backend:
  - legacy: assert ✔;
  - SC: `examples/configs/grpo_math_1B_megatron_generation_colocated_single_controller.yaml` inherits `cluster.gpus_per_node: 2` from its base, so it needs an override to 1.
- That path needs Megatron-Bridge and TransformerEngine built for sm_120.

**sm_120 blockers** (verified)
- `docker/Dockerfile:204-205` sets `TORCH_CUDA_ARCH_LIST="9.0 10.0"` ("H100 and B200"); `:172` sets `NVTE_CUDA_ARCHS=90;100`.
- Megatron raises if `TORCH_CUDA_ARCH_LIST` is unset (`lm_policy.py:263-267`).
- `3rdparty/{Automodel,Gym,Megatron-Bridge}-workspace/*` are **empty** in our checkout ✔. Run `git submodule update --init --recursive` first; even the default automodel path needs it.

**Dependencies:** Python `>=3.13.14,<3.14`, `vllm==0.29.0` ✔ (`pyproject.toml:167`), torch 2.13 cu130, `ray>=2.58`, DeepEP/DeepGEMM git builds (sm_90/100 oriented; they can be dropped for a dense 0.6B model).

### 2.12 Minimum bridge interface

**A. Native env.** A `@ray.remote` `EnvironmentInterface`, registered in `ENV_REGISTRY` and `ACTOR_ENVIRONMENT_REGISTRY` (`environments/utils.py:140-144`; `ray_actor_environment_registry.py:64-80`). It is text-in/text-out: NeMo-RL owns tokens and logprobs. There is **no hook for an env that does its own generation.**

**B. NeMo-Gym** (the only built-in full-episode token path)
- A resources server with `verify(BaseVerifyRequest) -> BaseVerifyResponse(reward, mask_sample)`, plus optional `/seed_session` and `/close_session` (`gym:nemo_gym/base_resources_server.py:92-110, 235-302`).
- Alternatively a Gymnasium-style `/reset` + `/step` server (`gym:resources_servers/gymnasium/base.py:72-105`) driven by `gymnasium_agent`.
- **All model calls must go through Gym's policy-model proxy** with token-id echo (`gym:nemo_gym/openai_utils.py:169-189` `TokenIDLogProbMixin`).
- **The contiguity invariant must hold.**
- NeMo-RL side: `env.should_use_nemo_gym: true`, `max_rollout_turns: null`, and no stop strings or top_k (`rollouts.py:2651-2683`).

---

## 3. SkyRL (3d0ae22)

**Layout.**
- The repo is now one `skyrl/` package. `skyrl-train/README.md` says the old code lives in `skyrl/{backends,train,utils}`.
- `skyrl-gym/` holds environments; `skyrl-agent/` holds agents (OpenHands CodeAct, ReAct).
- `skycap/` captures trajectories from unmodified OpenAI-client harnesses.
- `examples/train_integrations/` has **harbor, harbor_skycap, verifiers, openenv, openreward, modal**.
- **SGLang is gone at this pin**: only vLLM (`skyrl/train/config/config.py:1217-1218`). There is no DeepSpeed.

### 3.1 Process topology

**Driver.** `main()` → `ray.init` → `skyrl_entrypoint` as `@ray.remote(num_cpus=1)` (`skyrl/train/entrypoints/main_base.py:410-438`). Build order: inference client → generator → `RayPPOTrainer` → policy/critic/ref workers (`main_base.py:250-353`).

**Training workers**
- Each role is a `PPORayActorGroup` with a `(dp, sp)` mesh (`skyrl/backends/skyrl_train/workers/worker.py:122-137, 734-971`).
- A ref model exists only when KL is used; a critic only when `critic.model.path` is set (`trainer.py:695, 742-756`).

**Inference**
- Each engine is a `VLLMServerActor`: vLLM's OpenAI FastAPI app under uvicorn, with `enable_scale_out=True` for `/inference/v1/generate` (`inference_servers/vllm_server_actor.py:113-127, 694-765, 714`).
- SkyRL adds routes `/reset_prefix_cache`, `/fetch_weights`, `/skyrl/v1/load_lora_adapter` and `/skyrl/v1/generate`.

**Router.** The Rust `vllm_router` runs in a child process, default policy `consistent_hash` (`vllm_router.py:100-141`; `inference_servers/utils.py:442-446`).

**Client.** `RemoteInferenceClient` has a data plane through the router and a control-plane fan-out (pause, resume, sleep, wake, weights) to every server (`remote_inference_client.py:1-45, 1194-1217`).

**Generator and env.** These run in the trainer process. Env `init`/`step` run in a `ThreadPoolExecutor(max_env_workers=32)` (`skyrl_gym_generator.py:199-204, 286-296`; `config.py:1424`).

**Placement**
- `colocate_all=True` (default, `config.py:666`): one PACK PG; policy, ref and critic at 0.2 GPU each on the engines' bundles. Policy GPUs must equal rollout GPUs. Engines sleep after startup (`main_base.py:106-130, 243-246`; `trainer.py:708-756`).
- `colocate_all=False`: inference gets its own PG; policy and ref share a PG at 0.75/0.25 (`trainer.py:758-822`).
- External engines: `run_engines_locally=false`. PD disaggregation is available (`config.py:1319-1331`).

### 3.2 Training parallelism

- **FSDP2** (`distributed/fsdp_strategy.py`) or **Megatron**: TP, PP, CP, EP and ETP (`config.py:492-520`). `fsdp2` is an alias of `fsdp`.
- **Ulysses SP** only, per role (`config.py:693, 734, 755`).

**Batch keys**

| Key | Default | Meaning |
|---|---|---|
| `train_batch_size` | 1024 | Prompts per step |
| `policy_mini_batch_size` | 256 | Prompts per optimizer step |
| `micro_train_batch_size_per_gpu` | 1 | Gradient-accumulation split |
| `update_epochs_per_batch` | 1 | PPO epochs |
| `n_samples_per_prompt` | 5 | Samples per prompt |

Cites: `config.py:1349, 1536-1558`. Validation is `validate_batch_sizes` (`utils.py:86-146`).

**Loop.** For each epoch, for each mini-batch: `forward_backward`, then `optim_step` (`trainer.py:1612-1621`). The micro-batch loss is scaled by `len(micro)/len(mini)` (`worker.py:1012-1027`).

**Token-budget micro-batching.** `max_tokens_per_microbatch` uses a balanced bin-packer and pads micro-batch counts to match across DP (`worker_utils.py:265-293, 458-474`).

**Packing.** `remove_microbatch_padding=True` gives the THD layout and **requires flash-attn 2** (`config.py:1580`; `workers/model_wrapper.py:13, 151-156`).

**Old logprobs.** `recompute_old_logprobs_per_minibatch=True` makes the old-logprob forward pass use the training packing (`config.py:1572-1577`).

### 3.3 Rollout parallelism

**Engine shape.** `num_engines`, TP, PP, DP and EP all default to 1 (`config.py:1215-1239`). `max_num_seqs=1024`; prefix caching is on (`config.py:1246, 1286`).

**Cache salt.** `use_cache_salt=True` salts with `weight_version`, incremented after every sync ✔ (`worker_dispatch.py:873-875`).

**Load balancing.** Router-side. The client semaphore is `SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=512` × servers, with 30 data-plane retries (`remote_inference_client.py:97, 562-583`).

**Sticky sessions**
- `session_id = f"{instance_id}_{repetition_id}"`, sent as `X-Session-ID` (`skyrl_gym_generator.py:384-386`; `remote_inference_client.py:400-402`).
- `/finish_session` is called in a `finally` (`skyrl_gym_generator.py:742-743`).
- **[docs]** `sticky_least_loaded` is recommended for agentic RL (`docs/content/docs/getting-started/inference_architecture.mdx:47-60`).

### 3.4 Async design

**Modes**
1. Sync (default; `trainer.py:345-499`).
2. One-step off-policy example, which also asserts non-colocated ✔ (`examples/train/async/async_trainer.py:24`).
3. **Fully async** (`skyrl/train/fully_async_trainer.py`).

**`FullyAsyncConfig`** ✔ (`config.py:1029-1069`)

| Key | Default |
|---|---|
| `max_staleness_steps` | 4 |
| `num_parallel_generation_workers` | 768 |
| `sample_full_batch` | False (async-native DAPO filter) |
| `clear_kv_cache_on_weight_sync` | False (stale KV is reused) |
| Simulated-trainer knobs | — |

**Constraints** (`fully_async_trainer.py:353-409`)
- `mini ≤ workers ≤ mini·(staleness+1)`.
- `train_batch_size == mini`.
- No dynamic sampling.
- **`colocate_all` must be False** ✔ (`:392`).
- The loss must be rollout-anchored: `rollout_is`, `dppo`, or `cispo` with anchor=rollout.

**Batch formation.** N workers each pull one prompt, acquire a staleness slot, and generate a group into `asyncio.Queue(maxsize=mini·(staleness+1))`. The trainer takes `mini` groups (`:500-623, 922-999`).

**Staleness control.** AReaL-style capacity: `min(max_concurrent − running, (max_staleness + step)·mini − (accepted + running))` ✔ (`:181-187`). The bound is aggregate, and late groups are accepted with a warning. Each group records `global_step_when_scheduled` (`:957-976`). Metrics: `async/staleness_*` (`:1160-1164`).

**Weight update**
- `pause_generation()` in **KEEP** mode, broadcast, `resume_generation()`. There is no abort; in-flight turns continue under new weights ✔ (`worker_dispatch.py:859-871`).
- Optional KV offload during the sync (`:831-848`).

**Dynamic sampling (sync only):** `filter` (DAPO) or `replace` (POLARIS), `max_sample_batches=30` (`config.py:794-803`; `trainer.py:1657-1717`). `zero_variance_filter` loss-masks zero-variance groups.

### 3.5 Weight sync

**Backend selection** ✔ (`weight_sync/__init__.py:37-49`)
- `nccl` + `colocate_all` → **CUDA IPC**.
- `nccl` otherwise → broadcast from trainer rank 0.
- `delta` → XOR checkpoint deltas via shared filesystem, `gs://` or `s3://`.
- `sharded_rdt` → NIXL pull.

All run on vLLM's `WeightTransferEngine` with `packed=True` and a reusable 1 GB buffer (`weight_transfer_threshold_cuda_ipc_GB`, `config.py:1223`; `weight_senders.py:243-329`).

**Resharding.** FSDP all-gathers DTensors one at a time; Megatron uses Bridge `export_hf_weights`.

**Colocated sequence** (`worker_dispatch.py:732-822`)
1. Sleep the engine.
2. Load the policy and offload the optimizer.
3. `wake_up(weights)`.
4. IPC transfer.
5. `wake_up(kv_cache)`.
6. Offload the policy.

Sleep is level 2, or level 1 with LoRA.

**LoRA.** FSDP writes PEFT safetensors to `lora_sync_path`, then `load_lora_adapter` (`fsdp_worker.py:137-230`). FP8 sync is Megatron-only.

**Timing.** Logged as `timing/sync_weights` and `sync_weights_only_transfer`; no numbers are stated.

### 3.6 Environment integration

**skyrl-gym** (`skyrl-gym/skyrl_gym/envs/base_text_env.py:9-100`)
- `BaseTextEnv.init(prompt) -> (conversation, dict)`.
- `step(action: str) -> BaseTextEnvStepOutput{observations, reward, done, metadata, postprocessed_action}`.
- `close()` and metrics.
- Constructor `(env_config, extras)`; gym-style `register(id, entry_point)`.
- Synchronous only, run in threads.

**Agent loop** (`skyrl/train/generators/skyrl_gym_generator.py:339-743`)
1. `env.init`, then the chat template.
2. Each turn:
   - generate token ids with `session_id` and `cache_salt`;
   - `env.step(text)`;
   - observation ids via the "fixed base conversation" slice (`:213-231, 783-818, 1243-1310`).
3. Response tokens are appended verbatim (TITO).

**TITO breaks** in two cases: `postprocessed_action` (warns) and a custom chat template (retokenizes everything) (`:387-390, 561-572, 1200-1241`).

**Masking and rewards**
- Loss mask 1 on generated tokens; 0 on observations and synthetic EOS (`:132-146`).
- Rollout logprobs are padded with 0.0 over observations (`:148-158`).
- Per-step rewards are placed at each turn's last token (`:745-781`).
- `step_wise_trajectories` emits one training row per turn (`config.py:1391-1395`).
- `max_turns` is enforced **by the env** (`generator.max_turns=1`, `config.py:1354`).

**Failures** ✔
- `agent_loop` has no try/except and no timeout. `tqdm.gather` runs without `return_exceptions` (`:1023-1029`), so **one env exception fails the step**.
- Tool timeouts exist only inside the specific python and sql tools.
- A trainer-wide `step_timeout_s` exists.

**Harbor** (`examples/train_integrations/harbor/harbor_generator.py`)
- The agent's `api_base` is the router `/v1` (`:241, 264-272`).
- `collect_rollout_details=True` returns per-turn prompt ids, completion ids and logprobs (`:274-278`).
- Each attempt gets a uuid session.
- Retries (`:354-463`):
  - **timeouts are loss-masked**;
  - context-length errors get reward 0;
  - a missing verifier result is retried.
- Requires step-wise mode.

**mini-swe-agent.** Retokenizes and passes `rollout_logprobs=None`, so it is not TITO (`mini_swe_generator.py:204, 283`).

**verifiers.** `verifiers_generator.py:120-121` maps `completion_mask` → `loss_masks` and `completion_logprobs` → `rollout_logprobs`. It appears to target the legacy verifiers `processed_outputs` API; v1 compatibility is not verified.

**skyrl-agent.** A `SkyRLAgentGenerator(GeneratorInterface)` that calls `AutoAgentRunner` (`skyrl-agent/integrations/skyrl_train/skyrl_train_main.py:26-49`). It pins an OpenHands fork and `skyrl` from git main.

### 3.7 Losses, mismatch, advantages

**Policy losses** (`skyrl/backends/skyrl_train/utils/ppo_utils.py:446-1096`, registry `@register_policy_loss`):
- `regular`, `dual_clip`, `gspo`, `cispo` (anchor old or rollout), `clip_cov`, `kl_cov`, `sapo`, `cross_entropy`, `importance_sampling`;
- `rollout_is`: GLM-5 style; IcePop-like hard zero outside `(1−ε_l, 1+ε_h)`, vs rollout logprobs (`:824-863`);
- `dppo` ✔ (`:866-915`): binary-TV or **binary-KL** mask vs rollout logprobs, which is the **KPop-style** binary-KL mask from the survey.

**`off_policy_correction`** (`config.py:874-898`; `off_policy_correction_utils.py`)
- Token TIS capped at 2.0, or sequence TIS capped at 5.0.
- Sequence masks: `product` [0.5, 2] or `geometric` [0.99, 1.01] (MIS-style).
- Outlier-token rejection.

**Logprobs**
- vLLM returns `processed_logprobs`; the trainer logs `policy/rollout_train_logprobs_abs_diff_*` (`trainer.py:1450-1479`). This is Crucible's `rollout_probs_diff`.
- The default recomputes old logprobs (`trainer.py:1438-1446`). Rollout-anchored losses skip it (`trainer.py:1349-1374`).

**Loss reduction.** `token_mean` (default), `token_mean_legacy`, `sequence_mean`, `seq_mean_token_sum_norm` (Dr.GRPO), `prompt_mean` (`config.py:957-972`).

**Advantage estimators** (registry ✔ `@register_advantage_estimator`, `ppo_utils.py:518`):
- `grpo` (std normalisation on by default; **sums token rewards** ✔ `:1367`);
- `gae` (critic, value_clip 0.2);
- `rloo`, `reinforce++`, `maxrl`.

**KL.** `use_kl_loss=True`, coefficient 0.001, k3 estimator (`config.py:909-918`), or KL-in-reward.

### 3.8 Long tail

What exists:
- Fully async with the staleness manager.
- KEEP pause.
- Sticky sessions.
- Concurrency caps.
- DP and PD inference.
- Token-balanced micro-batches.
- A rate limiter.

**Not found:** a per-trajectory straggler timeout or abort in `SkyRLGymGenerator`.

### 3.9 Fault tolerance and resume

- Checkpoint layout: `global_step_N/{policy,critic}`, `data.pt`, `trainer_state.pt`, with the `latest_ckpt_global_step.txt` marker written last (`trainer.py:1728-1830`). `resume_mode=latest`; `ckpt_interval=10`.
- Fully async adds `fully_async_state.pt` (consumed UIDs). In-flight rollouts are regenerated on resume.
- **No Ray actor restart**: a dead worker fails the run.

### 3.10 Key config defaults

Defaults live in the dataclasses in `skyrl/train/config/config.py`; there is no YAML, and the CLI takes OmegaConf `key=value` overrides.

| Key | Default |
|---|---|
| `policy.model.path` | Qwen2.5-1.5B-Instruct |
| `colocate_all` | True |
| `strategy` | fsdp |
| `advantage_estimator` | grpo |
| `policy_loss_type` | regular |
| `loss_reduction` | token_mean |
| `eps_clip` | 0.2/0.2 |
| `use_kl_loss` | True |
| `n_samples_per_prompt` | 5 |
| `max_generate_length` | 1024 |
| `gpu_memory_utilization` | 0.8 |
| `weight_sync_backend` | nccl |
| `env_class` | gsm8k |
| `logger` | wandb |

### 3.11 One RTX 5090

**Sync colocated runs out of the box**
- ✔ `NUM_GPUS=1 bash examples/train/gsm8k/run_gsm8k.sh` (`examples/train/gsm8k/run_gsm8k.sh:11-30`): `colocate_all=true`, fsdp, Qwen2.5-1.5B-Instruct, `gpu_memory_utilization 0.8`, `$@` passthrough.
- Data: `uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k`.
- `examples/train/turn_level_rewards/run_gsm8k_multi_turn.sh` is **already `NUM_GPUS=1`**: Qwen2.5-0.5B-Instruct, multi-turn env, `batched=false`.

**Suggested small override** (untested; satisfies `validate_batch_sizes`):

```
NUM_GPUS=1 LOGGER=console bash examples/train/gsm8k/run_gsm8k.sh \
  trainer.policy.model.path=Qwen/Qwen3-0.6B trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=32 trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=16 trainer.eval_batch_size=64 \
  generator.inference_engine.gpu_memory_utilization=0.6
```

- Qwen3 `chat_template_kwargs` (e.g. `enable_thinking`) need `batched=false` ✔ (`skyrl_gym_generator.py:233-237`).
- **Async on one GPU is not available** (assert at `fully_async_trainer.py:392`).

**Dependencies.** Python ≥3.12; `--extra fsdp` brings vllm==0.30.0, torch 2.13 (cu130), a NovaSky combined FA2+FA4 flash-attn wheel whose pyproject comment says it embeds `sm_80 90 100 120` cubins, flashinfer, vllm-router, nixl, and ray==2.57.0 (`pyproject.toml:52, 117-135, 398-447`). **[docs]** CUDA 13.0 and driver ≥ r580; image `novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0` (`docs/content/docs/getting-started/installation.mdx:8, 33-35`). No sm_120 blocker was found in the code.

### 3.12 Minimum bridge interface

1. **skyrl-gym env.** Subclass `BaseTextEnv` and `register`. Text level: no black-box TITO, and the env must not raise.
2. **Custom Generator.** `GeneratorInterface.generate(GeneratorInput) -> GeneratorOutput` ✔ (`skyrl/train/generators/base.py:40-69`):
   - required: `prompt_token_ids`, `response_ids`, `rewards` (scalar or **per-token**), `loss_masks`;
   - `rollout_logprobs`, required for rollout-anchored losses and async;
   - `trajectory_ids` and `is_last_step` for step-wise;
   - optional timings, `rollout_expert_indices`, `rollout_sample_support`.

   Wire it in by overriding `BasePPOExp.get_generator` (`main_base.py:132-153`). Output order must match input order. A failed rollout can be emitted with `loss_mask=[0]`, as Harbor does (`harbor_generator.py:115-125`).
3. **External harness on the OpenAI endpoint.**
   - Point it at `client.get_endpoint_url()/v1` with a stable `X-Session-ID` and `finish_session`.
   - Recover tokens one of three ways: retokenize (not TITO), harness-reported ids and logprobs (Harbor `collect_rollout_details`), or **skycap token mode** in front of `/inference/v1/generate` (`skycap/README.md:12-60`).

---

## 4. Comparison table

| Dimension | prime-rl | NeMo-RL | SkyRL |
|---|---|---|---|
| Orchestration | Standalone processes; no Ray | Ray (virtual cluster, per-actor venvs); legacy driver or SingleController | Ray; trainer process drives the generator |
| Trainer ↔ rollout data | ZMQ PUB/SUB or filesystem msgpack `MicroBatch` | Ray object store, or TransferQueue (SC) | In-process `GeneratorOutput` → Ray object store |
| Env transport | ZMQ env servers (verifiers v1) | Ray env actors; NeMo-Gym HTTP servers (subprocesses) | In-process threads (skyrl-gym); HTTP for Harbor and other harnesses |
| Trainer backend | FSDP2 only (CP, EP, HSDP); no TP or PP | DTensor/FSDP2 (no PP) or Megatron (TP, PP, CP, EP, SP) | FSDP2 (+ Ulysses) or Megatron (TP, PP, CP, EP) |
| Mini-batches / PPO epochs | No; 1 optimizer step per batch | gbs-based; PPO epochs for PPO | Yes (`mini_batch`, `update_epochs`) |
| Packing | FFD + FLOP-balanced KK across DP | 6 packing algorithms or dynamic batching | Token-budget balanced micro-batches; THD padding removal (needs FA2) |
| Inference | vLLM ≥0.31 + vllm-router | vLLM 0.29 / SGLang / Megatron / TRT-LLM / Dynamo | vLLM 0.30 + vllm-router (SGLang removed) |
| Sticky multi-turn | Router `sticky_least_loaded` + `X-Session-ID` = trace id | Gym proxy session hash only; native path none | Router `consistent_hash` (or `sticky_least_loaded`) + `X-Session-ID` |
| Cache salt by version | Yes (version at group start) | `recompute_kv_cache_after_weight_updates` option | Yes (`weight_version`, incremented each sync) |
| Time-sliced colocation | **No** (disjoint GPUs) | Yes (sleep/wake, IPC refit); **sync only** unless Megatron generation | Yes (default; sleep/wake, IPC); **sync only** |
| Async knob | `TARGET_LAG=1` (fixed) + `max_off_policy_steps=8` | `max_trajectory_age_steps=1` (+ in-flight updates); SC `max_lookahead_versions` / windowed `max_staleness_versions` | `max_staleness_steps=4`, `num_parallel_generation_workers` |
| Staleness stamp | Per episode `PolicySpan(start, end)`; drop by oldest version | Per group (version, target); per call in SC token capture | Per group `global_step_when_scheduled`; aggregate bound |
| In-flight on update | `pause(mode=keep)` → partial rollout; stale groups cancelled first | keep (with `in_flight_weight_updates`) or drain; SC aborts out-of-window | `pause` KEEP; no abort |
| Weight sync | NCCL (default, ≥2 GPUs), filesystem (LoRA / no inference), NIXL | IPC/ZMQ (colocated), NCCL packed, `nccl_reshard`, NIXL, sparse delta | IPC (colocated), NCCL, delta (fs/gs/s3), NIXL `sharded_rdt` |
| Old logprobs | **Never recomputed** (ratio vs μ) | **Always recomputed** (prev) + IS vs gen | Recomputed, except rollout-anchored losses |
| Mismatch menu | IPO (\|Δp\| mask), IcePop, CISPO, custom loss; fp32 LM head default; sampling-mask replay | TIS, IcePop, seq-mask-tis (MIS-like), seq logprob-error filter; fp32 LM head option | Token/seq TIS, product/geometric seq mask (MIS), outlier reject, `rollout_is` (IcePop/DIS), **DPPO binary-KL (KPop-like)** |
| Objectives | GRPO (no std), max_rl, RAE, OPD/OPSD, echo, SFT-distill; **no critic** | GRPO, DAPO, GSPO, CISPO, RLOO, GDPO, R++, **PPO+critic (GAE)**, DPO, RM, SFT, distill | PPO, dual-clip, GSPO, CISPO, SAPO, clip/kl-cov, DPPO; GRPO/RLOO/R++/maxrl/**GAE+critic** |
| Loss aggregation | Global token-mean or prompt-mean | Token-mean or seq-mean | `token_mean`, `sequence_mean`, `seq_mean_token_sum_norm` (Dr.GRPO), `prompt_mean` |
| Dense per-token credit | Wire has per-token `advantages`; algo registry closed | Estimator dispatch hard-coded | Per-token `rewards`; open estimator and loss registries |
| Env failure | Excluded from baseline (not zeroed) | Infra vs data classes; SC retries | One exception fails the step (Harbor example masks timeouts) |
| Checkpoint | DCP trainer + `progress.pt`; buffer lost | `step_N` + optional replay buffer; SC recovery ledger | `global_step_N` + dataloader; async consumed UIDs |
| 1-GPU sync | No (launcher); manual space-slicing | **Yes** (`grpo_smoke.yaml`, Qwen3-0.6B) | **Yes** (`NUM_GPUS=1`, 1-GPU multi-turn script) |
| 1-GPU async | **Yes, via space-slicing** (untested) | Megatron generation only (sm_120 risk) | No |
| sm_120 risk | Low (5090 named as tested in README; FA2 wheels with sm_120) | **High** (Docker arch list 9.0/10.0; DeepEP/TE builds; empty submodules) | Low–medium (wheel claims sm_120; CUDA 13 / r580 driver) |
| License | Apache-2.0 (verifiers MIT) | Apache-2.0 (Gym Apache-2.0) | Apache-2.0 (skyrl-gym MIT) |

---

## 5. Recommendations for Crucible M12 and M11

The labels follow ARCHITECTURE §3: **DEPEND** (import it in `interop/` or `train/bridges/`), **PORT** (vendor with a license notice; all candidates are Apache-2.0 or MIT, so a PORT needs the header plus a `THIRD_PARTY.md` entry), and **PATTERN** (reimplement the idea).

### 5.1 Decision needed: what "colocated async on one GPU" means (DESIGN §10.3 point 2)

**Finding.** With vLLM, time-sliced colocation is synchronous in every framework (§TL;DR 1).

**Recommendation.** Define Crucible's MVP topology as **space-sliced colocation**:
- vLLM stays resident at a fixed memory fraction (about 0.3–0.4 on 32 GB);
- the trainer runs on the same GPU with optimizer CPU offload;
- weights move through a filesystem broadcast.

This gives real async (lag ≥ 1, version-capped staleness, keep-mode partial rollouts) without sleep/wake cost. Keep time-sliced sync colocation as the fallback mode. DESIGN §10.3 point 2 and ARCHITECTURE M12 `topology.py` should state this, since the exit gate says "colocated async GRPO run on the lab RTX 5090".

### 5.2 Trainer choice for the M12 exit gate

**Primary async run: prime-rl, DEPEND, through Path B or Path A.**
- **Path A** (planned in DESIGN §10.1): the `export verifiers` Taskset runs inside prime-rl's orchestrator. This tests the M11 export and the trainer at once. Dense credit, however, needs a custom algorithm, and `ALGORITHM_CLASSES` is a closed dict ✔ (`src/prime_rl/orchestrator/algo/__init__.py:43-52`). Options:
  - (i) a 10-line monkeypatch that registers `crucible_dense` before `build_algorithm`, and calls `assign_advantages(trace, per_token)` ✔ (`algo/routing.py:17-39`);
  - (ii) write `MessageNode.advantages` in the env (verifiers `graph.py:138`) and run a pass-through algorithm.

  Either way this is DEPEND plus one shim in `crucible/train/bridges/prime_rl.py`.
- **Path B** (also DEPEND): Crucible's M10 runtime **is** the orchestrator.
  - Convert each valid `crucible.rollout/v1` record into `TrainingSample`s ✔ (`src/prime_rl/transports/batch/types.py:31-80`):
    - one sample per compaction segment;
    - the segment's prompt deltas plus completions concatenated into `token_ids`;
    - `mask` True on completions;
    - `logprobs` = `Turn.logprobs` (0.0 on context);
    - `temperatures` from `Turn.sampling`;
    - per-token `advantages` from the M6 turn credit.
  - Pack with `prepare_batch`, write `rank_0.bin` through the filesystem batch transport, and drive the `broadcasts/step_N` handshake into a Crucible-managed vLLM.
  - This keeps M10's journal, blame, masking and staleness rules in Crucible (`staleness.py` applies the τ rule on `oldest_version()`), and uses prime-rl only as an FSDP2 trainer.
  - **Pin the prime-rl commit**: the msgspec structs are positional (`array_like=True`).
- **Losses.** Crucible's `mismatch.py` menu can be supplied as a prime-rl **custom loss** (`cfg:trainer.py:588-595`; `loss.py:47-61, 347-356`) taking `LossInputs`, so there is no fork. prime-rl's "no π_old recompute" is the DIS setting, which DESIGN §10.3 point 4 already lists.

**Second trainer for "one GRPO step per advantage mode": SkyRL, DEPEND.** Its open registries are the cheapest plugin surface:
- `@register_advantage_estimator` ✔ (`skyrl/backends/skyrl_train/utils/ppo_utils.py:518`) for the three M12 advantage modes;
- `@register_policy_loss` for the mismatch variants it lacks;
- a custom `GeneratorInterface` (`skyrl/train/generators/base.py:40-69`) that reads `crucible.rollout/v1` records.

It runs single-GPU sync colocated out of the box, and has **GAE + critic** for the DESIGN §10.3 point 5 critic mode (single-feedback enterprise workflows), which prime-rl lacks. ARCHITECTURE M12 currently names "prime-rl and verl". I recommend recording SkyRL as the critic-mode trainer, or as a substitute for verl if the verl fork's 1-GPU story is worse (not assessed here).

**NeMo-RL: TARGET for M11 only for now, not an M12 training dependency on the 5090.** Reasons:
- the sm_120 build risk ✔ (Dockerfile arch list);
- empty submodules in our checkout ✔;
- no async on one GPU without Megatron generation;
- hard-coded estimator dispatch ✔ (`grpo.py:2628-2664`).

Re-evaluate on the H100/B200 class.

### 5.3 What to PORT or PATTERN into `crucible/train/`

| Crucible module | Take | Source (file:line) | Mode |
|---|---|---|---|
| `staleness.py` | Oldest-version drop rule, queue time included; cancel stale in-flight groups *before* pausing for the update; a stale cancel voids the whole group | prime-rl `cfg:orchestrator.py:591-592`; `orchestrator/utils.py:42-61`; `dispatcher.py:401-439`; `train_sink.py:230-241` | PATTERN |
| `staleness.py` | AReaL-style admission capacity `(S + step)·mini − (accepted + running)` | SkyRL `fully_async_trainer.py:181-187` | PATTERN (a formula) |
| `staleness.py` | Windowed sampler with over-sampling and age-out eviction (the MiniMax windowed FIFO) | NeMo-RL `algorithms/async_utils/staleness_sampler.py:308-362` | PATTERN |
| `topology.py` | `pause_generation(mode="keep")` around weight load; a per-version `cache_salt` (per *call*, not per group, if Crucible wants no stale-KV reuse) | prime-rl `inference/vllm/server.py:40-44`; SkyRL `worker_dispatch.py:859-875` | PATTERN |
| `topology.py` | Filesystem weight handshake (`.sender_ready` / `.receiver_ready` / `.finished`, keep N and N−1) | prime-rl `transports/weights/base.py:17-108` | DEPEND (Path B) or PATTERN |
| `topology.py` | Adaptive in-flight cap on KV usage (AIMD), cancelling youngest groups first | prime-rl `orchestrator/concurrency.py:1-79`; `dispatcher.py:242-269` | PATTERN |
| `mismatch.py` | Token/sequence TIS, product/geometric sequence masks, outlier rejection | SkyRL `off_policy_correction_utils.py:24-190` | PORT (Apache-2.0, small and self-contained) or PATTERN |
| `mismatch.py` | KPop-style binary-KL mask (DPPO) | SkyRL `ppo_utils.py:866-915` | PORT or PATTERN |
| `mismatch.py` | IcePop / seq-mask-tis definitions and the `is_oob_ratio` global-normalisation trick | NeMo-RL `algorithms/loss/loss_functions.py:720-818` | PATTERN |
| `mismatch.py` | Sampling-mask replay (renormalize over the top-p/top-k support; fail if missing) | prime-rl `trainer/rl/loss.py:73-91`; `train_sink.py:263-271` | PATTERN. Also add `sampling_support` to `Turn` (see §6). |
| `objectives.py` | Loss-reduction menu incl. Dr.GRPO `seq_mean_token_sum_norm` | SkyRL `ppo_utils.py:1145-1221`; `config.py:957-972` | PATTERN |
| `objectives.py` (critic mode) | GAE with a critic, value clip, length-adaptive λ | SkyRL `ppo_utils.py` (gae); NeMo-RL `advantage_estimator.py:360-640` | PATTERN. Skip-observation GAE and cross-segment GAE remain Crucible's own (none of the three has them). |
| `monitors.py` | `rollout_train_logprobs_abs_diff_*` (= `rollout_probs_diff`), `async/staleness_*`, `off_policy/*`, `is_oob_ratio` | SkyRL `trainer.py:1450-1479`; prime-rl `orchestrator.py:677-692`; NeMo-RL `loss_functions.py:523-529` | PATTERN (metric names) |
| `credit.py` | Loss mask = 1 only on engine-generated assistant tokens (prompt-history assistant turns 0) | NeMo-RL `grpo.py:2222-2254`; SkyRL `skyrl_gym_generator.py:132-146` | PATTERN (already Crucible's rule) |
| `credit.py` | Failure semantics: exclude infra-failed rollouts from the group baseline, not zero them | prime-rl `orchestrator/algo/base.py:11-18`; `train_sink.py:243-255` | PATTERN. Matches M10 "masked, never 0"; Crucible must also recompute group statistics after the exclusion. |

### 5.4 M11 bridges

- **`export verifiers` → prime-rl (DEPEND).** Target the **v1** API (`vf:verifiers/v1/taskset.py`, `env.py`, `task.py`). The `IsolatedVerifierEnv` path stays as planned.
  - Two prime-rl facts the export must respect:
    1. **Env-server requests are untimed** ✔ (`vf:verifiers/v1/serve/client.py:101`). Set `env.timeout.episode` / `env.timeout.finalize` explicitly from Crucible's task limits (`vf:verifiers/v1/configs/env.py:22-27`), or a hung World stalls a group forever.
    2. **Errored traces are excluded, not zeroed** (`orchestrator/algo/base.py:11-18`). Crucible's `agent_broke_env` (a policy failure, scored 0) must therefore return a *scored* trace with reward 0, not an error, or the policy failure gets masked.
  - **Cross-check:** SkyRL also consumes verifiers environments (`examples/train_integrations/verifiers/verifiers_generator.py:120-121`), but via the legacy `processed_outputs` API. Do not rely on it for v1.
- **`export nemo_gym` → NeMo-RL (TARGET).**
  - The resources server returns `reward`, `mask_sample` and the `reward/<component>` columns, which must sum to the scalar (`rollouts.py:3196-3216`).
  - **The contiguity invariant ✔ (`environments/nemo_gym.py:1219-1228`) is Crucible's `prefix_break` rule.** A Crucible episode with policy-written compaction (DESIGN §10.4 point 3) **cannot** go to NeMo-RL as one row. Export one Gym row per segment, or disable compaction for NeMo-RL runs.
  - Reasoning stripping in the chat template also breaks contiguity. NeMo-RL's vLLM splice (`vllm_worker_async.py:961-973`) only repairs re-templating, not truncation.
  - Gym's model calls must use `vllm_model_for_training.yaml` (`return_token_id_information: true`).
- **`export openenv` / skyrl-gym → SkyRL (PATTERN for the env, DEPEND for the generator).**
  - A `BaseTextEnv` adapter is text-level: SkyRL owns templating, the env must not raise (no isolation ✔, `skyrl_gym_generator.py:1023-1029`), and the env enforces `max_turns`.
  - Prefer a **custom Generator** over Crucible rollout records for black-box harnesses, or SkyRL's own Harbor generator (`examples/train_integrations/harbor/harbor_generator.py`), since Crucible tasks are Harbor directories (DESIGN §4.3). Its retry policy (timeouts masked, context overflow scored 0, missing verifier retried, `:354-463`) matches M10's blame taxonomy.
- **Token capture for opaque harnesses.** Three patterns now exist:
  - verifiers interception + `bridge_to_next_turn` (`vf:verifiers/v1/clients/train.py:410`);
  - NeMo-Gym token-id echo + NeMo-RL prefix splice;
  - SkyRL **skycap** token mode in front of `/inference/v1/generate` (`skycap/README.md:12-60`; Apache-2.0).

  Crucible's capture proxy (M10, `crucible:capture_incomplete`) should keep producing `crucible.rollout/v1`, and each bridge converts from it. skycap is the closest PATTERN for a standalone proxy.

### 5.5 `rollout.py` mapping (contract check)

| `crucible.rollout/v1` | prime-rl `TrainingSample` | SkyRL `GeneratorOutput` | NeMo-Gym output item |
|---|---|---|---|
| Per-segment concat of `Turn.prompt_token_ids` (delta) + `completion_token_ids` | `token_ids` (one sample per segment / branch) | `prompt_token_ids` (first prompt) + `response_ids` (rest) | `prompt_token_ids` + `generation_token_ids` per assistant item (contiguous) |
| Completion = trainable | `mask` | `loss_masks` | Implicit (assistant items) |
| `Turn.logprobs` | `logprobs` (0.0 on context) | `rollout_logprobs` (0.0 on observations) | `generation_log_probs` |
| `Turn.sampling.temperature` | `temperatures` (per token) | Global `sampling_params` only | Request params |
| `Turn.advantage` (M6) | `advantages` (per token) ✔ | Per-token `rewards` at turn ends (`base.py:43`) + a custom estimator | `reward` scalar only; per-turn credit needs a NeMo-RL patch |
| `Turn.policy_version` | `PolicySpan` (episode) only | Group step only | Per call in SC token capture |
| `reward=None` / `mask_sample` | Error trace → excluded | `loss_mask=[0]` row | `mask_sample: true` |
| `prefix_break` | New branch = new sample | Breaks TITO (custom template path) | Assert fails |

**Gaps in Crucible's record:**
1. A top-p/top-k **sampling support / mask** per token (prime-rl hard-fails without it when top_p < 1; SkyRL has `rollout_sample_support`).
2. The **logprobs mode** (processed vs raw) as rollout metadata. All three use `processed_logprobs`, i.e. after temperature and filtering.

Both are small additions to `Turn` and should land with M12.

---

## 6. Pitfalls

1. **`mode="keep"` makes partial rollouts silently.**
   - In all three, in-flight requests continue under new weights after a pause.
   - **Mid-generation version tracking.** Crucible's `Turn.policy_version` stamped at capture time is the version at *request start*. A call that straddles an update was generated by two versions. Record `policy_version_end` (or the max) as well, and apply τ to the oldest.
   - **Stale KV.** With per-group (prime-rl) or not-recomputed (SkyRL default, NeMo-RL default) caches, KV computed under old weights is reused, so the logprobs do not exactly match any single policy. This argues for always logging `rollout_probs_diff` and keeping a correction on.
2. **Ratio semantics differ by trainer.** prime-rl's ratio is π_θ/μ (staleness + mismatch). NeMo-RL clips prev/π_θ and IS-weights prev/gen. SkyRL depends on the loss. Crucible's `mismatch.py` must declare which denominator each correction expects, and the bridges must feed `old` vs `rollout` logprobs accordingly. Otherwise a "TIS 2.0" means different things on different trainers.
3. **Failure semantics differ.**
   - prime-rl drops errored traces from the baseline.
   - SkyRL's generator crashes the whole step on one env exception.
   - NeMo-RL's legacy async per-sample loop **swallows** generation exceptions and breaks (`rollouts.py:1370-1372`).

   The bridge must convert M10 blame into each trainer's "masked" form *before* handing over. Never let a Crucible policy failure become a trainer-side error, which would mask it, and never let an infra failure become reward 0.
4. **Untimed env calls.** prime-rl's env-server requests are untimed, and NeMo-RL's sync path blocks on `ray.get` with no timeout (`rollouts.py:812-843`). Crucible's World must enforce its own wall-clock limit and return a *masked* result rather than hang.
5. **Rewards summed across turns** in NeMo-RL's native loop (`rollouts.py:1043-1061`), and **GRPO summing token rewards** in SkyRL (`ppo_utils.py:1367`). Feeding Crucible's per-step ΔΦ as per-step rewards is fine for the outcome sum, since Φ telescopes to the final reward. But GRPO then normalises on the sum, so the dense signal only reaches the tokens through a custom estimator. Do not expect the default GRPO to use the shaping.
6. **Observation tokenisation without a chat template** in NeMo-RL's native loop (`rollouts.py:1066-1076`) produces token streams that differ from what a chat-templated inference call would see. Use the Gym path for TITO-faithful multi-turn.
7. **Compaction versus contiguity.** Any trainer that assumes one contiguous token stream per rollout (NeMo-Gym's assert, SkyRL's TITO path) needs Crucible segments exported as separate samples. Group statistics must stay on whole rollouts (DESIGN §10.4 point 3), so the bridge must broadcast the rollout's advantage to every segment sample rather than normalising per segment.
8. **The 1-GPU memory model differs.**
   - SkyRL and NeMo-RL time-slice: vLLM sleeps (level 2 / level 1) and models are offloaded per phase, so every step pays wake/offload cost.
   - prime-rl space-slicing never sleeps vLLM, so `gpu_memory_utilization` must be set *before* the trainer starts. vLLM grabs memory at init, so start vLLM first.
   - NeMo-RL Automodel CPU offload raises at world_size=1 (`models/automodel/setup.py:535-540`).
9. **prime-rl NCCL broadcast needs ≥2 GPUs** ✔ (`cfg:rl.py:288-296`). On one GPU use the filesystem broadcast. That writes full HF safetensors every step (about 1.2 GB for 0.6B bf16), so put the broadcast directory on local NVMe.
10. **Version and build drift.** The pins are vLLM 0.31 (prime-rl), 0.29 (NeMo-RL) and 0.30 (SkyRL); torch 2.13 cu130 everywhere; CUDA 13 drivers (≥ r580) are required. One venv per trainer: they cannot share an environment. prime-rl's `deps/verifiers` submodule (`da75527`) differs from our verifiers checkout (`ae819de`), so pin the verifiers version in the M11 export test to what prime-rl runs.
11. **Docs disagree with code.**
    - prime-rl: "single-GPU supported" vs the 2-GPU launcher; the FA4 docstring vs FA2 on SM12x; LM-head chunk 1024 vs 8192; the "drains" comment vs `mode=keep`.
    - SkyRL README: SGLang and CUDA 12.8, both stale.
    - NeMo-RL's `GenerationRouter` docstring calls Gym routing round-robin, while the code hashes the session.

    Cite code, not docs, in DESIGN.
12. **Licensing.** Everything is Apache-2.0 or MIT, so PORT is allowed with headers. `skyrl-agent` declares a license file that does not exist, and it pulls an OpenHands fork, so do not PORT from `skyrl-agent`.
