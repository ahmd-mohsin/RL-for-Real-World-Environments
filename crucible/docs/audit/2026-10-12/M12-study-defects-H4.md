# M12 study H4: defects in the scale-out training path and the framework bridges

Scope: `C/train/{trainer (DistributedTorchTrainer, offload/onload, state_dict), fsdp, spmd, cluster, placement, launch, state, hf}.py`, `C/train/bridges/*`, and the launch entry the GPU scripts actually run (`packs/_toy/src/crucible_pack_toy/m12/gate_dist.py`, `scripts/modal_gate_dist.py`, `scripts/modal_calibrate.py`, `scripts/modal_infra_check.py`, repo `scripts/repacss.sh`, `scripts/lab_gpu.sh`). Tests read: `tests/unit/test_fsdp.py` (+ `_fsdp_worker.py`, `_fsdp_probe.py`), `test_train_bridges.py`, `test_train_offload.py`, `test_train_state.py`, `test_cluster.py`, `test_launch.py`.

Paths: `C` = `crucible/src/crucible/`. `$S` = the session scratchpad `m12/h4/`. The probes are:

| probe | what it is |
|---|---|
| `worker.py` | one gloo rank per scenario |
| `launch.py` | runs W ranks with a wall timeout |
| `sweep_parity.py` | the parity sweep |
| `placement_check.py` | placement plans on H100 clusters |
| `single_checks.py` | single-process checks |
| `bridges/repro_*.py` | the bridge reproducers |

Run from `crucible/` with `PYTHONPATH=src .venv/bin/python $S/<probe>`. Distributed probes take arguments like this: `$S/launch.py <scenario> <world> '<json>' <timeout_s>`.

Method: read-only study on HEAD `3b522cb`. CPU only: gloo process groups with 2, 4 and 8 ranks as separate processes, torch 2.14.1, transformers 5.19, offline. **[ran]** means a probe reproduced it. **[read]** means the finding comes from the code alone. No GPU, NCCL, Slurm or upstream framework install was available. The bridge checks therefore ran each upstream framework's own source, extracted unchanged from the local checkouts: verl 5eb4826, TRL 72cf37d, SkyRL 3d0ae22, rLLM 4e828c8. No repo file other than this document was changed.

## 0. What holds (checked, no defect)

**Parity with the single-process update** (`sweep_parity.out`, 130 runs) [ran]:
- Coverage: `DistributedTorchTrainer`, `FSDPTrainer` in step and layer modes, and HSDP 2×(W/2) in step and layer modes, at world sizes 2, 4 and 8. Each was run with:
  - dis and PPO with π_θold recomputed;
  - a skewed batch (uneven micro-batch counts, so lockstep padding runs);
  - `grad_clip=1e-3`, so clipping is active on every step;
  - `updates_per_batch=2`;
  - the k3 KL term on;
  - `reset_optimizer` between steps;
  - `offload("cpu")`/`onload()` between steps.
- Result: every case equals `TorchTrainer` on the full batch after 2 updates. The worst max |Δθ| is 1.7e-5 (the reset case at lr 1e-2). Elsewhere it is ≤ 1e-5.
- The pre-clip grad norm matches to 1.5e-8. So `clip_grad_norm_` sees the **global** norm, both over all-reduced replicas and over DTensor shards on a 2D mesh.
- Replicas stay bit-identical.
- The only failures are the zero-item cases (V1).

**Planner HSDP meshes** [ran] (`hsdp41_step.out`, `hsdp81_step.out`, `hsdp24_layer.out`): the shapes the planner emits, (4,1), (8,1) and (2,4), run and match the reference with clipping active (max |Δθ| ≤ 1.1e-6).

**DCP resharding** [ran] (`dcp_*.out`): a checkpoint saved at 4 ranks loads exactly (parameters and AdamW moments, Δ = 0) in each of these cases:
- at 2 ranks;
- at 8 ranks;
- from an HSDP 2×2 save into 1D FSDP at 2 ranks;
- from a layer-mode save into a step-mode trainer.

**Tied embeddings under FSDP** [ran] (`tied_*.out`): a Llama with `tie_word_embeddings=True` under `HFPolicy`, with blocks as units, in both layer and step modes:
- HF key set equal;
- max |Δθ| ≤ 4.8e-7;
- `lm_head.weight` is still tied after the update.

**HFPolicy checks** [read]:
- Right padding without `position_ids` is correct.
- Gradient checkpointing defaults to `use_reentrant=False` in transformers 5.19 (`modeling_utils.py:3137`).
- The fp32 LM head is applied outside autocast.

**Bridges: what still matches upstream** (bridge agent) [ran]:
- verl `register_adv_est` exists (`core_algos.py:116`). The estimator name is a free string (`algorithm.py:734`). The call signature and `(adv, returns)` return match `ray_trainer.py:251-279`.
- TRL still uses `env_mask`, and a `None` reward is excluded from the baseline.
- rLLM `Step` accepts every emitted key, and `use_precomputed_advantage` exists.

## 1. Findings

| id | sev | file:line | defect | evidence |
|---|---|---|---|---|
| V1 | P0 | `C/train/trainer.py:645-648,655-656` | **A rank that receives no items hangs every distributed trainer.** The per-step metric vector is built from rank-local keys. A rank whose partition is empty all-reduces `[loss, n_mb]` (2 floats) while its peers all-reduce 2+K floats. All 23 runs with fewer items than ranks hang or crash, on every trainer kind and at W = 2, 4 and 8. The trigger is any batch with fewer trained items than DP ranks: critic mode n=1 with small B, heavy masking, or a few long segments at dp=8/16. | `sweep_parity.out` (all `few_items`/`one_item` rows); `few4t.out`: ranks 0-2 reduce 8 values, rank 3 reduces 2, then gloo times out [ran] |
| V2 | P0 | `C/train/trainer.py:712-716`; `C/train/fsdp.py:123-131` | **A parameter that one rank's forward does not reach breaks the gradient reduction.** `_reduce_grads` flattens only grads that are not None, so the buckets differ in size across ranks. The result depends on the trainer: <ul><li>`DistributedTorchTrainer`: hangs, then aborts.</li><li>FSDP layer mode: hangs.</li><li>FSDP step mode: no error, but every parameter is updated wrongly (max |Δθ| = 0.02 = 2·lr against the reference, on all 9 tensors).</li></ul> Realistic triggers: `VisionHFPolicy` on a rank whose items have no images (the vision tower is unused), per-expert MoE parameters (custom or R3 MoE, pre-fused HF MoE), and conditional heads. | `moe_unused` scenario: `moe_ddp.out` `[TIMEOUT,-6]`; `moe_fsdp_layer.out` gloo recv timeout; fsdp_step `per_key` all 0.02. Control with no routed item: 2.5e-5 (V16) [ran] |
| V3 | P0 | `C/train/state.py:113-117`; `C/train/spmd.py:37-81` | **Resume through `SPMDDriver` with a replicated trainer diverges the replicas.** The non-sharded restore calls `tr.model.load_state_dict` / `tr.opt.load_state_dict` on the driver only, and `SPMDDriver` has no command that loads state. The followers keep their fresh init and fresh AdamW, so after the first post-resume step the replicas differ (spread 0.074 after 2 steps). Separately, `tr.updates = …` sets an attribute on the driver object, because `SPMDDriver` forwards reads but not writes. The driver then reports a frozen `updates` (4) while the trainer counts from 0 (2; the correct value is 6). The next checkpoint's `state.pt` stores the stale value. The FSDP resume test reads that shadow (`_fsdp_worker.py:194`, `test_fsdp.py:83`), so it cannot catch this. | `resume_ddp.out`: `spread_after_run1=0.0`, `spread_after_resume_run=0.0741`, `driver_updates_attr=4`, `trainer_updates_attr=2` [ran] |
| V4 | P0 | `packs/_toy/.../m12/gate_dist.py:65-68`; `C/train/trainer.py:702-704`; `C/train/placement.py:221-233` | **Constructing the trainer for a model the planner accepts runs out of GPU memory.** The only shipped multi-GPU entry loads the full model in fp32 on every GPU (`HFPolicy.from_pretrained(dtype=float32, device="cuda")`). `DistributedTorchTrainer.__init__` then broadcasts every full parameter before `fully_shard`. The planner accepts Qwen3-30B-A3B on 2×8 H100 at 48.8 GB/GPU sharded (mesh (1,10)), but the load peak is 122 GB per GPU. `train_mesh` never checks the load peak. | `placement_check.out` rows for `Qwen3-30B-A3B`: `fp32_load_peak_gb: 122.0` against 80 GB [ran, planner; the OOM itself is from reading the code] |
| V5 | P1 | `C/train/spmd.py:27-34`; `gate_dist.py:92,163-167` | **Followers can time out before the driver joins.** Followers call `follow()` → `command_group()` at once. The driver creates its `SPMDDriver` only after waiting for engine health (up to 900 s per engine) and running the held-out "before" eval. The new gloo group's rendezvous waits only the **default PG timeout** (NCCL default 10 min), not the 7-day timeout passed to `new_group`. Whenever engine start plus eval takes longer than that (an 8B model at scale), the followers die and the driver then hangs. | `late_driver` (PG timeout 15 s, driver 40 s late): follower `Timeout waiting for key: default_pg/0//1//cpu//0/rank_0 after 15000 ms`; driver killed at 90 s (`ld.out`) [ran] |
| V6 | P1 | `C/train/spmd.py:27-102`; `C/train/cluster.py:120-126` | **SPMD has no liveness check.** <ul><li>A follower that dies mid-step leaves the driver blocked for the full PG timeout: 120 s of 120 s, because gloo never noticed the dead peer (`fd.out`).</li><li>A driver that dies leaves the followers waiting on the 7-day command group (still blocked at the 90 s kill, `dd.out`).</li></ul> On Slurm the allocation holds 8 or 16 GPUs until walltime. `LocalLauncher` waits on each process in turn with the full timeout. No script sets `TORCH_NCCL_ASYNC_ERROR_HANDLING` except `modal_infra_check`. | `follower_dies`, `driver_dies` [ran] |
| V7 | P1 | `C/train/placement.py:295-302`; `C/train/fsdp.py:209-214` | **The planner chooses colocated layouts that `FSDPTrainer` cannot run.** For agentic workloads it picks `colocated_partial` for 0.6B through 30B. Its cost assumes the trainer state is swapped to host over PCIe, but `FSDPTrainer.offload("cpu")` raises `NotImplementedError`. Without offload, trainer plus engine plus KV memory comes to 88 / 105.6 / 109.4 GB per 80 GB GPU (4B, 8B, 30B-A3B). | `placement_check.out` (`colocated_resident_gb`); `single_checks.out` [1] [ran] |
| V8 | P1 | `C/train/cluster.py:73-76` | **`LOCAL_RANK` does not index the visible GPU.** It is set to the physical GPU index while `CUDA_VISIBLE_DEVICES` holds that one GPU. torch's convention (DeviceMesh, torchrun) is `set_device(LOCAL_RANK)`, which is an invalid ordinal on 14 of 16 ranks of a 2×8 plan. It works today only because `gate_dist.entry` calls `torch.cuda.set_device(0)` first. Any other entry, including `FSDPTrainer`'s default `init_device_mesh`, fails on node-local ranks ≥ 1. | `single_checks.out` [4]: torch `device_mesh.py:499-505` calls `set_device(1)` with 1 visible GPU, which fails with `invalid device ordinal` [ran, CUDA handle faked] |
| V9 | P1 | `C/train/cluster.py:49,107-136,146-178`; `tests/unit/test_cluster.py:106-114` | **No multi-node launch path exists.** <ul><li>`LocalLauncher` starts every node's specs on the current host. For 2×8 that is 16 processes, with each of GPUs 0-7 claimed twice.</li><li>`RayLauncher` copies `MASTER_ADDR` from the specs (default `127.0.0.1`), fixed before the placement groups land. Ranks on node 1 rendezvous with themselves.</li><li>Nothing maps Slurm (`SLURM_PROCID`/`SLURM_LOCALID`/`SLURM_NODEID`, master from the nodelist) or torchrun.</li><li>The "two-node" test runs on one host with gloo on loopback. It shows rendezvous, not placement.</li></ul> | `single_checks.out` [5] [ran]; Ray [read] |
| V10 | P1 | `C/train/placement.py:264-286`; `C/train/cluster.py:60-63` | **Multi-GPU engines get one GPU per process.** Elastic and colocated GPUs are one process each with one visible GPU, but the engine size can be > 1. For example, Qwen3-30B-A3B on 2×8 (elastic) gives elastic engine 0 as two processes, each with `CUDA_VISIBLE_DEVICES` of one GPU and `ENGINE_GPUS=2`. A TP-2 vLLM engine cannot start from either. | `placement_check` layout (elastic engine 0 = node 1 GPUs 0 and 1, separate processes) [ran] |
| V11 | P1 | `C/train/trainer.py:690-693`; `C/train/fsdp.py:58-61,73-77` | **Critic and full-vocabulary OPD cannot run on more than one process.** `DistributedTorchTrainer` and `FSDPTrainer` accept no `critic` and no `teacher`. Critic mode is the n=1 enterprise route. The layer-mode refusal names `full_vocab` as if step mode supported it, but constructing it raises. | `single_checks.out` [2] [ran] |
| V12 | P2 | `C/train/trainer.py:694-704,725` | `src` is used both as a group rank (`self.rank == self.src`) and as a global rank (`broadcast(src=…)`). On a trainer subgroup `[2,3]`: <ul><li>`src=0` raises `Global rank 0 is not part of group`;</li><li>`src=2` means no rank sends the batch, which ends in `TypeError: 'NoneType' object is not iterable`.</li></ul> | `subgroup_src0.out`, `subgroup_src2.out` [ran] |
| V13 | P2 | `C/train/spmd.py:61-63` | `SPMDDriver.offload(mode, path=…)` drops `path`, both locally and for the followers. NVMe offload goes to `tempfile` (`/tmp`, often small and node-local on clusters). The `mkstemp` fd also leaks (`trainer.py:159`). | `single_checks.out` [3]: `[('nvme', None)]` [ran] |
| V14 | P2 | `C/train/placement.py:149-151`; `gate_dist.py:67-68` | `trainer_kwargs()` is not constructor-ready. `FSDPTrainer` rejects `mesh_shape` and `dp`; `TrainerConfig` rejects `mesh_shape` and `reshard`. The only entry ignores it and builds a 1D mesh, so the planner's HSDP shape (e.g. (2,4), (4,4)) is never applied. | `placement_check.out` (`fsdp_rejects`, `trainerconfig_rejects`) [ran] |
| V15 | P2 | `C/train/placement.py:260-261`; `C/train/runtime.py:76` | `env_workers` counts the CPU slots of every node (384 on 2×8). The runtime turns that number into threads inside the rank-0 process on one node. | `placement_check.out` [ran] / [read] |
| V16 | P2 | `C/train/trainer.py:645-648`; `C/train/fsdp.py:128` | The distributed paths give a parameter with no gradient anywhere a zero tensor (empty rank, FSDP probe). The single process leaves it None. AdamW then applies weight decay and moment decay only in the distributed run. | `moe_unused` with `no_route`: FSDP-step Δ = 2.5e-5 per step, DDP 6e-8 [ran] |
| V17 | P2 | `C/train/trainer.py:709-716`; `C/train/fsdp.py:82-91` | Memory and bandwidth costs at scale: <ul><li>DDP all-reduces one flat bucket, which is a second full copy of every gradient (+32 GB for an 8B model in fp32).</li><li>FSDP has no `MixedPrecisionPolicy`, so it all-gathers and reduce-scatters fp32: twice the bytes of bf16 on every layer gather.</li><li>The per-parameter init broadcast issues one collective per tensor.</li></ul> | [read] |
| V18 | P2 | `scripts/*`, repo `scripts/repacss.sh` | Gaps for the next venue (REPACSS, 8 nodes × 4 H100 NVL 94 GB): <ul><li>`ClusterSpec` defaults assume 8 GPUs per node, 80 GB and 450 GB/s all-to-all NVLink.</li><li>No calibration entry exists for H100 NVL.</li><li>No sbatch or srun recipe launches training.</li><li>Outside Modal no launch sets `NCCL_CUMEM_ENABLE=0` or an explicit `init_process_group(timeout=…)`.</li></ul> No secrets are printed. The Modal scripts keep results (`02-live-validation.json`, the `crucible-gate-runs` volume). | [read] |
| V19 | P1 | `C/train/bridges/adapters.py:22-26` | verl `AgentLoopOutput` requires `metrics` (`agent_loop.py:107`), which the adapter omits. | `bridges/repro_verl.py` [1]: `ValidationError … metrics Field required` [ran] |
| V20 | P1 | `C/train/bridges/verl_bridge.py:57-60` | `register_verl()` has no caller and there is no `verl.plugins` entry point. verl's trainer runs in a Ray actor (`main_ppo.py:102`), so the estimator is unknown there. | repro [3]: `Unknown advantage estimator simply: crucible_delta` [ran] |
| V21 | P1 | `C/train/bridges/verl_bridge.py:4-33`; `adapters.py:24-26` | The turn credit never reaches the estimator. verl builds `rm_scores` from the scalar `reward_score` at the last position (`agent_loop.py:143-148`), and `extra_fields` go to `non_tensor_batch`, which the estimator never receives. So `crucible_delta` runs as plain outcome GRPO, and DESIGN.md:1011 overstates it. | repro [2],[4]: advantages `[0.5,0.5,0,0,0.5]`, expected `[0.6,0.6,0,0,0.5]` [ran] |
| V22 | P1 | `C/train/bridges/trl_bridge.py:24-26` | The bridge does not match TRL `vllm_generation.generate`, which returns a 4-tuple whose logprobs have shape `(batch, len, num_logprobs)` and may be None (`vllm_generation.py:550-576`). | `repro_trl.py` [1]: `TypeError float() … 'list'` [ran] |
| V23 | P1 | `C/train/bridges/trl_bridge.py:36` | TRL already repeats prompts `num_generations` times (`RepeatSampler`, `grpo_trainer.py:1270`). The bridge runs N episodes per input, giving N² rows with duplicate seeds, and the reward count no longer matches. | repro [2] [ran] |
| V24 | P1 | `C/train/bridges/trl_bridge.py:40-50` | Masked episodes are dropped from the rows but kept in `crucible_reward`, so the rewards shift onto the wrong completions. | repro [3]: 3 rows, 4 rewards [ran] |
| V25 | P1 | `C/train/bridges/adapters.py:29-41` | The SkyRL `GeneratorOutput` is missing required keys, starting with `rollout_metrics` (`trainer.py:1069`). The row count also fails `validate_generator_output`: multi-segment and masked rollouts break `num_prompts == num_responses`, and step-wise mode needs `is_last_step`. | `repro_skyrl.py` [1],[3]-[5] [ran] |
| V26 | P1 | `C/train/bridges/adapters.py:50-56` | rLLM's verl backend reads only `step.model_output` and skips steps without it, so 0 rows are emitted. It also casts the advantage to `float`, and per-token lists fail there (`transform.py:289-320,593-598`; `buffer.py:238`). | `repro_rllm.py` [3]-[5] [ran] |
| V27 | P1 | `tests/unit/test_train_bridges.py:33-96` | The bridge tests are vacuous. They check only Crucible's own dict keys. They call `crucible_delta` with token rewards verl never delivers, replace `trl_gen`, assert the N² row contract, and include no masked case. | §3 |
| V28 | P2 | `C/train/bridges/verl_bridge.py:41-44` | R is read at the last *sampled* token, but verl writes it at the last response position. A trajectory ending in observation tokens scores 0. With `use_kl_in_reward`, −β·kl is read as turn credit. The TRL bridge ignores `max_tokens`/`temperature` and always reports `finish="stop"`. SkyRL `stop_reasons` is hard-coded. | repro [5],[6] [ran] |

## 2. Details, fixes and regression tests

**V1: a rank with zero items.**
- **Fix:** reduce a fixed-length vector on every rank. Either take the key set from the correction itself (constant per config), or exchange `keys` once with `all_gather_object` and fill missing entries with 0.
- **Better:** make "this rank has no items" the same path as padding. In layer mode `_equalize` already exists; extend it so the scalar reduction never depends on local data.
- **Test:** `parity` at W=4 with n=3 (and at W=8 with n=1) finishes within the timeout and equals `TorchTrainer`. It fails today with a gloo timeout.

**V2: unreached parameters.**
- **Fix:** before any gradient reduction, give every parameter that requires grad a gradient (`zeros_like` when None) on every rank. For FSDP step mode, do this on the unsharded parameters before the probe backward, or make the probe touch every parameter. Also add a debug check that all-gathers the per-rank "grad present" bitmask.
- **Test:** the `moe_unused` scenario (`Gated` model, expert reached only by rank 0's item) for ddp, fsdp_step and fsdp_layer. Each must finish and match `TorchTrainer` to 1e-5. Today: hang, 0.02 drift, hang.

**V3: resume with replicated trainers.**
- **Fix:** add a `load_state` command to `SPMDDriver`/`follow`. The driver broadcasts the blob, or every rank loads the shared `state.pt`. Alternatively, after `restore`, call a `DistributedTorchTrainer.sync_from(src)` that broadcasts parameters and optimizer state.
- Make `SPMDDriver.__setattr__` forward trainer attributes (`updates`).
- **Test:** `resume_ddp`: `spread_after_resume_run == 0` and `trainer.updates == driver.updates == 6`. Change `test_fsdp.py:83` to read `drv2.trainer.updates` after `rt2.run()`.

**V4: the load path runs out of memory.**
- **Fix:** build the model on the `meta` device on every rank, `fully_shard` it, then load the weights. Use `torch.distributed.checkpoint.state_dict.set_model_state_dict(..., StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))` with rank 0 reading from disk, or DCP-load HF safetensors shard by shard.
- Drop the init broadcast for sharded trainers.
- Add `fp32_load_peak` to `train_mesh` until then.
- **Test:** a planner unit test that a plan whose `4·params` exceeds GPU memory either reports a meta-init build or is refused.
- **Test:** a gloo test that `FSDPTrainer` accepts a meta-device model and loads a full state from rank 0.

**V5: followers time out waiting for the driver.**
- **Fix:** create the command group (`SPMDDriver`) right after `init_process_group`, before engine health checks or eval. Or `follow()` takes a group that `entry` built collectively at start.
- Pass `timeout=` to `init_process_group` in the entry.
- **Test:** `late_driver` with PG timeout 15 s and driver delay 40 s completes.

**V6: no liveness.**
- **Fix:** follower watchdog. A driver heartbeat key in the default store is polled from `follow` before each blocking receive, and the follower exits when the key goes stale.
- The driver uses `monitored_barrier` (gloo) with a short timeout before each command.
- Set `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` in the launchers.
- `LocalLauncher` kills the rest when one process exits non-zero, like `srun --kill-on-bad-exit=1`.
- **Test:** `driver_dies` and `follower_dies` end within 2× the heartbeat.

**V7: colocated plans `FSDPTrainer` cannot run.**
- **Fix:** implement `FSDPTrainer.offload("cpu")`. Move each DTensor's local shard and the optimizer state to pinned host memory (`p._local_tensor` / `to_local()`), and `onload` them back.
- Or have `plan()` reject colocated topologies for sharded trainers and add a resident-memory check (trainer + engine weights + KV ≤ GPU).
- **Test:** `plan(ClusterSpec(1), 8B, agentic)` never returns a placement whose resident sum exceeds `gpu_mem_gb`. `FSDPTrainer.offload/onload` round-trips bit-exact on gloo.

**V8: `LOCAL_RANK`.**
- **Fix:** with `CUDA_VISIBLE_DEVICES` restricted to one GPU, emit `LOCAL_RANK=0` (keep the physical index in `CRUCIBLE_GPU`). Or do not restrict and keep `LOCAL_RANK` as the GPU index.
- Ship one `train.cluster.init_rank()` helper that sets the device and initializes the process group, and use it in every entry.
- **Test:** for every trainer spec, `int(LOCAL_RANK) < len(CUDA_VISIBLE_DEVICES.split(","))`.

**V9: no multi-node launch.**
- **Fix:** a `SlurmLauncher`, or `process_specs_from_slurm()`, that maps `SLURM_PROCID`, `SLURM_LOCALID` and `SLURM_NODEID` onto specs and takes `MASTER_ADDR` from the first host in `scontrol show hostnames`. It should ship with an sbatch template (one task per GPU, `--kill-on-bad-exit=1`).
- `LocalLauncher(node=k)` should run only node k's specs.
- `RayLauncher` should first learn the node IP of rank 0's bundle and then set `MASTER_ADDR`.
- **Test:** `LocalLauncher` refuses multi-node specs without a node filter. A fake-Slurm-env unit test checks rank, world and master mapping for 2×4.

**V10: multi-GPU engines split across processes.**
- **Fix:** do not split an engine across single-GPU processes. For elastic and colocated GPUs with `engine_gpus > 1`, start one engine process per engine that sees all its GPUs, with the trainer ranks as separate processes. Or support only `engine_gpus == 1` there and make `plan` enforce it.
- **Test:** for every spec with `ENGINE_GPUS=k`, the spec (or the engine's process) sees k GPUs.

**V11: no critic or teacher in distributed trainers.**
- **Fix:** accept `critic`/`teacher` and broadcast the critic's init. Partition the critic data by rank like the policy items (today every rank would regress on the full batch), all-reduce the critic gradients, and reduce explained variance across ranks.
- **Test:** a W=2 gloo critic run equals the single-process critic update.

**V12: `src` semantics.**
- **Fix:** store `src` as a group rank and pass `group_src=` (torch ≥ 2.6) to `broadcast` and `broadcast_object_list`, or convert it with `dist.get_global_rank(group, src)`.
- **Test:** the `subgroup` scenario runs with the trainer on `[2,3]`.

**V13: dropped offload path.**
- **Fix:** forward `path` as `self._send("offload", mode, path)` and `self.trainer.offload(mode, path=path)`, and in `follow` call `trainer.offload(args[0], path=args[1])`. Close the `mkstemp` fd.
- **Test:** a recording trainer receives the path.

**V14: `trainer_kwargs` not usable.**
- **Fix:** add a `Placement.build_trainer(model, cfg)` helper (mesh via `init_device_mesh`, `dp` into `TrainerConfig`, `reshard` into `FSDPTrainer`), and use it in `gate_dist`.
- **Test:** construct from the plan's kwargs on gloo.

**V15: `env_workers` counted across nodes.** Cap it at one node's slots, or document it as cluster-wide and spread the env workers across nodes (env servers).

**V16: zero gradients versus None.** Make the single process and the distributed paths agree. Either zero-fill in `TorchTrainer` too, or leave None everywhere by excluding grad-less parameters consistently.

**V17: memory and bandwidth.**
- Bucket the DDP all-reduce (e.g. 256 MB), or use `torch.nn.parallel.DistributedDataParallel`'s reducer pattern.
- FSDP: `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`, which matches the existing autocast intent.

**V18: REPACSS readiness.**
- Add `ClusterSpec` presets per venue (REPACSS: `gpus_per_node=4, gpu_mem_gb=94`, NVL bridged pairs).
- Add a training sbatch script that sets the NCCL env baseline (`NCCL_CUMEM_ENABLE=0`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`) and keeps outputs on shared storage. `state.save` needs a filesystem the ranks share: multi-node DCP writes into the driver's `mkdtemp`, `state.py:76-80`.

**V19-V28: bridges.** Fixes are in the bridge agent's reproducers (`$S/bridges/`):
- **V19:** emit `metrics`.
- **V20:** a `verl.plugins` entry point whose module registers the estimator at import.
- **V21:** a Crucible AgentLoop and reward path that writes per-token `rm_scores`, or an estimator that receives `non_tensor_batch`.
- **V22:** pick the sampled token's logprob using `out[3]`.
- **V23:** run one episode per input with `seed = i % N`.
- **V24:** keep a placeholder row with `env_mask=[0]` and reward None.
- **V25:** emit every `GeneratorOutput` key and exactly one row per input (or step-wise ids).
- **V26:** set `model_output=ModelOutput(...)`, and pass a scalar advantage on the verl backend.

**Regression tests for the bridges:** a `test_train_bridges_upstream.py`, skipped when `RL-Environments/06-frameworks-and-env-code` is absent. It runs the upstream source unchanged:
- verl: `AgentLoopOutput.as_dict` and `compute_advantage`;
- SkyRL: `validate_generator_output`;
- rLLM: `_process_trajectory`;
- TRL: the 4-tuple `generate` and the `rewards_per_func` merge.

V20 is also open as M11 D9, and the TRL items overlap M11 D2 (`M11-sweep-G4.md`).

## 3. Tests that do not test what they claim

- **`test_fsdp.py`** (all scenarios) always uses 8-12 items at W ≤ 4. No case has a rank without items (V1) or a parameter reached by only some ranks (V2). The worker maps devices as `cuda:{rank}` (`_fsdp_worker.py:68`), the global-rank-as-device pattern that breaks on node 2. It runs only on Modal and is test-only, but it is the pattern the NCCL path was validated with.
- **`test_fsdp.py:83`** checks `resumed_updates == 8` by reading the `SPMDDriver` shadow attribute that `restore` writes (V3). It checks no counter after the resumed run, and it never resumes a replicated (non-sharded) trainer through the driver.
- **`test_cluster.py:106-114`** ("runs a two-node placement") starts both "nodes" on one host over loopback gloo with `CRUCIBLE_CPU_ONLY`. Node identity is only a label. The test cannot see V8, V9 or V10. The Ray test is the same (`:117-133`).
- **`test_train_bridges.py`**: see V27.
- **No test** builds `FSDPTrainer` from `Placement.trainer_kwargs()` (V14), runs a colocated plan with `FSDPTrainer` (V7), or starts followers before a slow driver (V5).

## 4. Not reproduced / out of reach here

- NCCL-specific behaviour: watchdog timing, `NCCL_CUMEM_ENABLE`, and P2P over H100 NVL bridges. No GPU was available.
- HFPolicy `shared_prefix_logprobs` passes a 4D float mask. Under `attn_implementation="flash_attention_2"` that mask is not honoured. Prefix sharing is refused only in layer mode, so step mode with flash-attn could compute wrong log-probs. Not run (no flash-attn on CPU).
- `BroadcastWeightSync.update(ranks=…)` (`weightsync.py:226-227`) creates a new process group (a new NCCL communicator) on every publish. That leaks communicators over a long run. Weight sync belongs to another hunt and is only noted here.
