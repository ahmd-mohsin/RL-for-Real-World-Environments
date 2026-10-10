# M12 `train` — the RL infrastructure layer

**Status: core built (2026-10-10).**
- **CPU end to end ✅:** every topology learns.
- **Checklist (52 items, `docs/research/m11-m12/06-article-systems.md` §8):** audited 2026-10-10. The earlier "implemented as tests where it applies" overstated it. Status table in §9: 30 done, 22 partial, 0 missing; the gaps are being closed in order.
- **GPU gate ⏳:** Qwen3-0.6B on the lab RTX 5090 is staged and waiting for the shared GPU.

- **Research** (`docs/research/m11-m12/`):
  - `01` verl + slime;
  - `02` AReaL + ROLL;
  - `03` prime-rl + NeMo-RL + SkyRL;
  - `04` checkpoint-engine + OpenRLHF + TRL + rLLM;
  - `05` interop and environment diversity;
  - `06` the systems content of the 2026 agentic-RL infra article (52 testable requirements).
- **Source pins:** `00-source-pins.txt`.
- **Code:** `src/crucible/train/`.
- **Tests:** `tests/unit/test_train_{math,sched,runtime,bridges}.py`.

## 1. Planes and contracts

| Plane | Object | Contract |
|---|---|---|
| Control | `Scheduler` (`scheduler.py`) | deterministic decisions over pool tags: admit, select at the sample barrier, evict; the same object runs in the simulator and the runtime |
| Data | `DataPool` (`pool.py`) | tags (status, group, source, submit index/version, versions touched, length, reward, cuts) separate from values (`crucible.rollout/v1`); scheduling reads tags only |
| Rollout | `Runtime` tasks + `EnginePool` (`runtime.py`, `engine.py`) | one asyncio task per rollout; model calls to engines (sticky crc32 routing); environment steps never wait for weights |
| Weight | `InprocPublisher` / `CheckpointPublisher` (`weights.py`); `DistributedPublisher` / `EngineReceiver` over `BroadcastWeightSync` (`weightsync.py`) | `publish()` → version + 1, exactly; engines load atomically between calls; every response carries its serving version; the distributed path: bucketed double-buffered broadcast, all-reduced error codes, atomic commit, subgroup updates (§6a) |
| Training | `TrainerBackend` (`trainer.py`) | `step(records, groups)`; `state_dict()`; reference `TorchTrainer`; bridges to verl / TRL / SkyRL / rLLM (`bridges/`) |
| Record | `crucible.rollout/v1` (`rollout.py`) | per call: token ids (linear delta storage), behaviour logprobs, `policy_version`, `version_spans`, sampling, credit; per rollout: reward (None = masked), segments, trajectory tokens; `valid_for_training()` |

## 2. Parallelization granularity: every unit, defined once

| Unit | Definition | Where it lives |
|---|---|---|
| **rollout** | one episode of one sample: the **dispatch unit** (DeepSeek-V4.1: batch-level dispatch oscillates, group-level dispatch waits for the in-group long tail) | `Runtime._rollout`, one asyncio task |
| **group** | n rollouts of one prompt: an **admission and advantage unit only**, never a barrier; GRPO advantage is computed when the group is terminal; n = 1 is critic mode (no groups) | `DataPool.Group`, `objectives.group_advantages` |
| **model call** | one engine request; the **version-purity unit**: served by one version (`call`), or resumed across versions with `version_spans` (`token`) | `GenRequest` / `GenResult`, `Turn` |
| **token** | the **loss unit**: corrections and aggregation are per token; observation tokens are never trained | `mismatch.correct`, `objectives.aggregate` |
| **segment** | a context-compaction segment: one causal training sequence; credit crosses segments by (γλ)^{N>s} | `Turn.segment`, `bridges.common.flatten`, `objectives.gae` |
| **global batch** | the groups one optimizer step consumes (B groups × n) | `SchedulerConfig.B` |
| **mini batch** | `updates_per_batch` slices of a global batch (default 1 update) | `TrainerConfig.updates_per_batch` |
| **micro batch** | one forward/backward on one device, bounded by a **token budget over forward tokens** (prompt + completion: what memory holds); gradients accumulate, each scaled by its share of the global **trained**-token count (exactly the full-batch gradient) | `batching.micro_batches`, `loss_scale` |
| **DP rank** | a data-parallel replica (a process: `DistributedTorchTrainer`); the global batch is partitioned so summed **forward** tokens (prompt + completion) balance (largest-first; imbalance ≤ 1.1 tested); gradients all-reduced in one flattened bucket; equals the full-batch step | `batching.balance_partitions`, `trainer.DistributedTorchTrainer` |
| **engine slot** | one continuous-batching sequence; `capacity` per engine | `Engine.capacity` |
| **env worker** | environment steps run off the GPU pool (CPU / sandboxes); an env step holds no engine slot | the World backends (M1), `CopyTask.latency` in tests |

## 3. Topologies and synchronization points

- **S1** is the sample barrier: the scheduler selects B groups.
- **S2** is the weight publish.
- **S3** is what happens to in-flight calls at a publish.

| Topology | S1 | Engines during training | S2 | S3 | Placement on one GPU |
|---|---|---|---|---|---|
| `colocated_sync` | the admitted batch is terminal (admission cap S = 1) | idle | after every step | none in flight | time-shared |
| `colocated_partial` (default) | B selectable groups | **aborted + paused** (vLLM `/pause?mode=abort` + `/sleep`), environments keep running | every `publish_every` steps | aborted calls re-issued whole on the new version (`call`), or resumed with their partial output (`token`, spans recorded) | time-shared (carry-over of unfinished rollouts only) |
| `disaggregated_stream` | B selectable groups | **keep generating** (trainer in a thread / another process) | every `publish_every` steps | in-flight calls finish on their pinned version; new calls get the new one | **memory split** (engine at a fixed fraction, trainer resident): real overlap |

Facts behind the table:
- No surveyed framework overlaps generation and training on one GPU with sleep/wake (research 02, 03). On a single GPU, real overlap requires the memory split.
- Weights are pinned at call start (`TorchEngine`, `VLLMServerEngine`), so a publish never changes weights under a running call.

## 4. Staleness: three enforcement points

1. **Admission.** At most `S × B × n` **untrained** rollouts, counting in-flight *and* buffered (B.16). An in-flight-only cap let simulated rollouts train at staleness 12 with τ = 2.
2. **Selection.**
   - Staleness = v_train − oldest version touched.
   - `wait` (default): block selection while any live rollout has staleness ≥ τ. Nothing is dropped; the stale work finishes and trains.
   - `drop`: evict terminal groups with staleness > τ; admission refills them.
   - Optional `max_offpolicy_frac`.
3. **Loss.** The correction, by default `dis(0.8, 3.0)`, needs only rollout logprobs, so it is valid under multi-version async. Corrections that need π_θold refuse to run without it.

**Buffer policies:**
- `oldest_first` (MiMo);
- `greedy`;
- `windowed_fifo` (MiniMax: only items with submission index < head + W, W = ⌈0.3·N⌉; the article's lag bound of 10 at batch 8, W 4 is tested).

**Length-bias controls:**
- per-source caps with a deficit-based mix;
- warmup batches;
- `drop_early_short`;
- a KS-distance monitor.

**Group backfill for masked members:** `pad`, `refill_member` or `drop_group`. Dynamic sampling drops zero-variance groups.

## 5. Mismatch, objectives, credit

- **Corrections** (`mismatch.py`), each a pure function with the article's numbers as golden tests: ppo (clip-higher, dual clip), tis, icepop, kpop, dis, mimo4, mispo, hispo, cispo, gspo.
- **Metrics logged every step:** `rollout_probs_diff`, r_dis quantiles, masked and clipped token shares. Without π_θold, `rollout_probs_diff` includes drift since sampling.
- **Aggregation:** token-mean global (default), seq-mean, prompt-mean, HisPO constant.
- **Advantages:** group mean, RLOO or std, with masked rollouts excluded. Critic GAE has skip-observation, cross-segment (γλ)^{N>s}, truncation bootstrap and length-adaptive λ. DAPO overlong shaping is available.
- **Credit:** M6 turn credit on each call's tokens; frozen sub-agents get loss mask 0; compaction summaries train with the episode advantage.
- **FP32 LM head:** `HFPolicy(lm_head_fp32=True)`, default on.

## 6. Environments, engines, trainers

- **Environments:**
  - the token-env contract (`envs.py`);
  - **Crucible Worlds** through `world_env.world_rollout`: `LLMToolPolicy` → `EngineLLM` → engine;
  - **exact TITO** (`chat.py`): the next prompt = prompt + sampled completion (+EOS) + only the rendered suffix of the new messages, so the model's output is never re-tokenized;
  - a rebuilt context (compaction) starts a new segment.
- **Engines:**
  - `TorchEngine` (tiny real policy, bf16 numerics);
  - `ScriptedEngine` (test double);
  - `VLLMServerEngine` (vLLM 0.31: token-id completions, `/pause`, `/sleep`, `/wake_up`, `/collective_rpc` → `CrucibleWorkerExtension.crucible_load_weights`, `/reset_prefix_cache`).
- **Trainers:**
  - `TorchTrainer` over `TinyPolicy` or `HFPolicy` (fp32 master + bf16 autocast, batched logprobs);
  - bridges: verl (`crucible_delta` estimator through `register_adv_est`, with turn credit encoded in token rewards), TRL (`rollout_func` over colocated vLLM), SkyRL `GeneratorOutput`, rLLM `Step` with precomputed per-token advantages.

## 6a. Distributed weight sync (`weightsync.py`; checkpoint-engine PATTERN)

| Step | Rule |
|---|---|
| Plan | gathered once; sorted (owner, name); 256 B aligned; one owner per bucket; tensor > bucket refused; portable JSON + crc |
| Bucket size | MIN over ranks of free × 0.9 → a third if the largest tensor fits (staging room), else half; ≤ `max_bucket`; ≥ largest tensor |
| Header | `[version, plan crc]` from the coordinator; a crc mismatch aborts every rank |
| Pipeline | 2 × bucket buffer; bucket g → half g mod 2 → broadcast from its owner; reload of g−1 on a worker thread overlaps; at most one reload in flight, so a half is never overwritten while read |
| Errors | after every bucket, `all_reduce(MAX)` of (failing rank + 1): every rank raises `WeightSyncError` naming the same rank and bucket |
| Commit | receivers stage into a fresh dict; `commit(version, state)` only after the last bucket succeeded everywhere; failure = old weights and version keep serving |
| Subgroups | `ranks=[…]`: only members transfer (a restarted engine catches up); non-members return at once |
| Runtime | `DistributedPublisher.publish()` (collective in a thread; the event loop keeps rolling out) ↔ `EngineReceiver.receive_once()` → `engine.load_weights(state, version)` |

## 7. Results

| Gate item | Result |
|---|---|
| Token-in-token-out round trip | ✅ prompts reconstruct exactly from delta storage; World episodes through the runtime have no prefix break (`test_world_episodes_through_the_runtime_with_exact_tito`) |
| Masking when a token record is missing | ✅ `valid_for_training`; train mode masks `capture_incomplete` (M10) |
| Staleness drop rule and windowed-FIFO bounds | ✅ GLM worked example, `wait` never drops and bounds τ, `drop` never trains over τ, windowed-FIFO lag ≤ 10 |
| Mismatch corrections on synthetic distributions | ✅ golden tests for every correction |
| Real data parallelism | ✅ DP = 2 gloo processes, 10 micro batches, `dis` and `ppo`: replicas bitwise identical, equal to the single-process full-batch update within 6e-7 |
| Distributed weight sync | ✅ real processes over gloo: bitwise across dtypes and buckets, slow-reload pipeline, sharded owners, all-rank abort with no commit, subgroup + elastic plan, publisher → receiver |
| Truncated and agent-broke-env cases | ✅ (M10 records; masked never trained, agent failures scored) |
| CPU end to end | ✅ a real tiny policy learns under sync (0.08 → 0.37), partial call-level (0.09 → 0.61, 906 aborts re-issued), partial token-level (0.09 → 0.57, resumed with spans), disaggregated (0.07 → 0.51) |
| One colocated async GRPO run on the toy pack on the lab RTX 5090, with staleness and `rollout_probs_diff` recorded | ⏳ `python -m crucible_pack_toy.m12.lab status` (staged; waiting for the shared GPU) |

## 8. Remaining

- the prime-rl algorithm bridge (mapping `steps.jsonl` rows to sampled nodes);
- a verl `CrucibleAgentLoop` on verl's server manager;
- an AReaL proxy bridge;
- zero-copy hand-off of received weights to a separate vLLM process (CUDA IPC) and RDMA P2P reads (the bucketed NCCL/gloo broadcast itself is built, §6a);
- a vLLM in-process engine (`AsyncLLM`) for colocated mode without HTTP;
- R3 routing replay and top-p candidate capture (MoE policies).

## 9. Checklist status (audit 2026-10-10, updated as items close)

| # | item | status | missing |
|---|---|---|---|
| 1 | four plane interfaces + version | partial | StateStore (checkpoint + policy state + paused-call registry); WeightPublisher protocol |
| 2 | publish +1, per-call version | partial | capture-proxy version stamping; 100-call mid-rollout test |
| 3 | call-level version purity | done | |
| 4 | version_spans gate | done | |
| 5 | publish_s, reprefill per step | partial | per-step `weights/publish_s`, `rollout/reprefill_tokens` keys |
| 6 | topologies, identical-loss golden | partial | cross-topology loss golden test |
| 7 | offload knobs | **done** | `TorchTrainer.offload(cpu\|nvme)`/`onload()` (moments preserved; CPU round trip bitwise); runtime offloads after each publish and onloads before each step (colocated; disaggregated refuses); measured on MPS: ≤ 5 % of the training peak stays resident; `kv_on_train` validated per engine (`VLLMServerEngine`: discard only) |
| 8 | separate env worker pool | **done** | `EnvWorkerPool` (`env_workers`); tests: blocking envs parallel, engines serve during a block, env steps finish while paused, World episodes on workers |
| 9 | per-phase bubbles | **done** | `idle/rollout_s`, `idle/train_s`, `idle/env_wait_s`, `env/busy_s`, `engine/utilization` per step |
| 10 | colocated_sync on-policy | done | |
| 11 | partial barrier semantics | partial | KV release; byte-identical World checkpoint test |
| 12 | disaggregated never pauses | done | |
| 13 | publish every K, optimizer reset | partial | tests for K > 1 and moment zeroing |
| 14 | warmup batches | done | test |
| 15 | per-source admission | done | lower-bound test |
| 16 | terminal-only GRPO, critic no barrier | partial | critic mode (n = 1) |
| 17 | group backfill | done | pad / drop_group tests |
| 18 | loss_agg + DAPO golden | done | |
| 19 | DP balancing | done | real DP: `DistributedTorchTrainer` |
| 20 | drop ≤ τ, one refill per evict | partial | refill-exactly-one assertion |
| 21 | wait, zero evictions | done | |
| 22 | in-flight ≤ S × B | done | |
| 23 | GLM example | done | |
| 24 | staleness metric set | partial | staleness_worst, evicted_staleness, admission_blocked_s |
| 25 | max_offpolicy_frac | done | |
| 26 | windowed FIFO rule | done | |
| 27 | Fig. 9 reproduction | partial | W = 1 ≡ FIFO, W = N ≡ greedy tests |
| 28 | windowed lag ≤ 10 | done | |
| 29 | length-bias monitor | partial | first-10-batch hard share; drop_early_short test |
| 30 | source mix ± 10 % | partial | sim mix test |
| 31 | mismatch metrics every step | done | |
| 32 | nine corrections | done | |
| 33 | correction goldens | done | |
| 34 | π_θold refusal on multi-version | **done** | `TorchTrainer` refuses an implicit π_θold for rollouts spanning versions; explicit `recompute_current` / `rollout` allowed; `test_trainer_refuses_implicit_old_logprobs_for_multi_version_rollouts` |
| 35 | FP32 LM head in the run record | partial | precision of both engines in the record |
| 36 | routing confidence, R3 | **done** | `TinyMoEPolicy` (top-k router; `forward(routing=)` replays the choice, gates keep gradient); engines record per-token experts (kept across resumes); `TrainerConfig.routing_replay="r3"` replays them and refuses < 100 % coverage; `moe/routing_confidence` logged; a full-bf16 engine flips ~11 % of routes and R3 more than halves the logprob gap on flipped tokens. `VLLMServerEngine` routing capture is open (vLLM's routed-experts output) |
| 37 | top-p candidates renormalization | **done** | `train/sampling.py` (one nucleus definition); engine records `GenResult.candidates`, kept across token-level resumes; `Turn.candidates` + `candidates_valid`; trainer renormalizes (TinyPolicy, HFPolicy); matches the engine within 1e-5; `VLLMServerEngine` refuses top_p < 1 until it reports candidates |
| 38 | post_update_transform / QDQ | partial | QDQ + bit-identical test |
| 39 | tag/value split | partial | remote-value never-fetch test; harness tag |
| 40 | async reward deadlines | **done** | `reward_fn`/`aux_fn` + `reward_deadline_s` (late = still pending), `reward_timeout_s` (masked `reward_timeout`), grader crash masked, `aux_deadline_s` → `credit_fallback`; `grading` status is untrained and not selectable; `test_train_reward_async.py` |
| 41 | blocked probe stays trainable | partial | train-path test |
| 42 | infra mask vs agent_broke_env | done | |
| 43 | harness per group | partial | per-group assignment, harness share |
| 44 | compaction params recorded | partial | defaults, run-record entry |
| 45 | summary calls carry advantage | done | direct assertion |
| 46 | cross-segment GAE | partial | critic trainer; 3-segment golden |
| 47 | skip-observation GAE | done | wired to a trainer |
| 48 | critic knobs | partial | TTUR, freeze attention |
| 49 | critical_steps, frozen sub-agents | partial | critical_steps; TorchTrainer frozen mask |
| 50 | prefix sharing | **done** | `TrainerConfig.prefix_sharing`: items with one full prompt in a micro batch encode it once (`TinyPolicy`: GRU state; `HFPolicy`: Forge-style packing, positions restart per completion, 4D *float* mask, since a bool mask is ignored by eager attention); loss and gradients equal the unpacked step within 1e-6; `train/prefix_tokens_saved`. Open: keeping a group in one micro batch when packing |
| 51 | per-step monitor set | partial | runtime feeds truncation, entropy, zero-variance, hacks |
| 52 | pass-rate alarm → re-audit | partial | re-audit hook |
