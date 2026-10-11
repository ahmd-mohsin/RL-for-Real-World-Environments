# M12 `train` — the RL infrastructure layer

**Status: core built (2026-10-10).**
- **CPU end to end ✅:** every topology learns.
- **Checklist (52 items, `docs/research/m11-m12/06-article-systems.md` §8):** audited 2026-10-10. The earlier "implemented as tests where it applies" overstated it. Status table in §9: 41 done, 11 partial, 0 missing; the gaps are being closed in order.
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
| Observability | `Tracking` over sinks (`sinks.py`; `RuntimeConfig.sinks`) | step metrics, every trained `crucible.rollout/v1` record, every failure and the final summary fan out to JSONL / TensorBoard / W&B (offline by default) / console; a sink that raises is disabled with one warning and listed in the summary (`sinks_failed`), never fatal (m13 C-1) |
| Group grading | `Runtime(group_fn=...)`, `RuntimeConfig.group_deadline_s`; operators in `credit/gar.py` | runs once a GRPO group is terminal (all n scored records), before advantages: may rewrite rewards and set `RolloutRecord.advantage` (the trainer uses it for a group whose scored members all carry one); starts at terminal time, gets what is left of the deadline at the barrier, else raw advantages + `group_credit_fallback`; a raising operator falls back and is counted as a harness failure; `group/graded`, `group/fallback`, `group/barrier_wait_s` (m13 C-2). GAR (mass-conserving, λ-capped redistribution of passing advantage by quality, hacks zeroed first) and GRS (`R_test·S_sol·S_beh`, rubric grounding check) are the first operators (C-3) |
| Hack defence | `world/intercept.py` (online, at `World._call_tool`), `credit/hack_detector.py` (two-level) | intercepted calls are never executed (refusal or dummy result; the episode continues and stays trainable); rollouts carry level-1 `flags` (intercepts, test/verifier writes, exit-status spoofing, test skipping); `detector_group_fn(judge)` asks the level-2 judge only about flagged rollouts and writes `hack/confirmed` (an unparseable or failed judge never confirms); chained before GAR it zeroes confirmed hacks; `confirmed_hack_rate` above `StepMonitor.max_confirmed_hack_rate` (2%) raises an alarm (m13 C-4) |
| Distillation | `TeacherPool` (`teachers.py`: `TorchTeacher`, `VLLMTeacher`), `Runtime(teacher_pool=...)`, `TrainerConfig.opd_*`, `kl_coef*` | a finished rollout is scored by its routed teacher (and a reference) before it enters the pool, off the training critical path, under `teacher_deadline_s` (miss: RL-only `opd_fallback`, or mask); vocabularies checked before any rollout; token OPD = dense advantage `rl_coef·A + β(log π_T − log π_behaviour)`, full-vocabulary OPD = `β·KL(π_θ‖π_T)` with a local teacher; k3 KL to the reference with per-source coefficients (m13 C-5) |
| Multi-domain | per-source maps on `Runtime` (`env_factory`, `harness`, `rollout_fn`, `reward_fn`, `aux_fn`: `{source: value, "*": default}`), `SchedulerConfig.max_staleness_by_source` | one run mixes token envs, World tasks and judge-graded sources; every record carries `source` and `timings.active_s`; per-source τ (groups at their bound train first; under `wait`, a group past its bound is evicted, never trained over it); `reward/mean/<src>`, `len/{mean,p99}/<src>`, `active_s/{mean,p99}/<src>`, `records/<src>` per step (m13 C-6) |
| Context modes | `ContextConfig.strategy` / `mode` (`actors/policy.py`), `EngineLLM(thinking=...)` + `codec.thinking_behaviour()` | strategies summary / keep_recent / discard_all / tiered / hybrid, each recorded as `segments[i].strategy`; `mode="action"`: a `context_compact(summary)` tool, the summary being the policy's own recorded call (trainable), no automatic trigger; thinking history preserved / interleaved / strip, probed from the chat template (Qwen3 = interleaved), a mismatched mode refused; a re-render marks `new_segment` and the policy starts a segment (m13 C-7) |
| Shaping | `credit/shaping.py`, `LimitPolicy` `penalty:<c>`, `TrainerConfig.tool_format_penalty` | incomplete-trajectory penalty on limit stops; malformed tool calls counted per model call (`Turn.format_errors`) and charged to that call's tokens only; speed term `1/(1+t/T)` added to passing rollouts only (`with_speed`); verbosity budget group stage (longer than σ·ℓ₀ loses); `grader/s`, `grader/usd`, `grader/share` per step (m13 C-8) |
| Model gateway | `GatewayPool`, `HTTPEngine`, `engine_app` (`gateway.py`) | drop-in `EnginePool` over local or remote engines: sticky per-rollout routing that queues up to `spill_factor × capacity` before spilling; a failing call returns `abort` (re-issued by the runtime) and evicts the engine, dropping only its sessions; health loop evicts and revives (after replaying the last weights); version = min over live engines (m13 D-1) |
| Remote rollout workers | `LeaseController` / `RolloutWorker` / `env_rollout` (`workers.py`) | episodes run in worker processes on any host; jobs are leased with heartbeats (own thread, so CPU-bound envs cannot starve them) and returned as records; model calls come back through `/gen` → `Runtime._call` (versions, barrier aborts, staleness unchanged); an expired lease is re-leased as a new attempt (pool tag reset) up to `max_attempts`, then masked; calls and results from a superseded attempt are refused, so nothing is trained twice; workers pull small batches (`prefetch`) so load spreads (m13 D-2) |
| Remote environments | `interop/nemo_gym/client.py` (`remote_episode`, `remote_world_rollout`), hardened `nemo_gym/server.py` | Worlds served by env-server processes over the NeMo-Gym resources protocol; any `Policy` (incl. `LLMToolPolicy` with exact TITO) drives them; session affinity by rollout id; servers bound live sessions (`max_concurrent_episodes`; a queue timeout is a 503 the client masks, never a 0), reap idle sessions, expose `/health` (m13 D-3) |
| Launcher | `train/launch.py` (`Launcher`, `run_job`, `python -m crucible.train.launch job.toml`) | one command starts engine servers behind the gateway, rollout workers, env servers and the controller; supervises children (restart on death, same port, up to `max_restarts`); kills every process group at the end; Ray-free (the Ray launcher, audit item G, is not built) (m13 D-4) |
| Engines | `HFGenerateEngine` (`hf_engine.py`), `capabilities()` on every engine and `EnginePool`, `probe_logprobs_mode` | a real HF model as an engine on CPU / MPS / CUDA with the TorchEngine contract (version pinned per call, token-level resume, top-p candidates, global and per-call abort, KV-cache decoding, publish into a copy while calls run); engines declare their capabilities (the pool offers what all offer); the logprob convention (processed vs raw) is measured, not assumed (m13 F-1/F-2) |

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
| One colocated async GRPO run on the toy pack on the lab RTX 5090, with staleness and `rollout_probs_diff` recorded | ❌→⏳ first run (2026-10-10): 0 steps, 228k masked rollouts (transformers-5 chat-codec bug, swallowed silently); fixed + failure accounting + circuit breaker; re-staged |

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
| 1 | four plane interfaces + version | partial | state plane built (`train/state.py`: barrier checkpoints, atomic + hashed, bitwise resume in sync mode incl. critic; in-flight rollouts restart); open: a formal `WeightPublisher` protocol class and mid-episode rollout resume (M10 World checkpoints) |
| 2 | publish +1, per-call version | partial | capture-proxy version stamping; 100-call mid-rollout test |
| 3 | call-level version purity | done | |
| 4 | version_spans gate | done | |
| 5 | publish_s, reprefill per step | **done** | `weights/publish_s`, `rollout/reprefill_tokens` (per-step delta) |
| 6 | topologies, identical-loss golden | **done** | one batch through each topology's `_train`: identical loss |
| 7 | offload knobs | **done** | `TorchTrainer.offload(cpu\|nvme)`/`onload()` (moments preserved; CPU round trip bitwise); runtime offloads after each publish and onloads before each step (colocated; disaggregated refuses); measured on MPS: ≤ 5 % of the training peak stays resident; `kv_on_train` validated per engine (`VLLMServerEngine`: discard only) |
| 8 | separate env worker pool | **done** | `EnvWorkerPool` (`env_workers`); tests: blocking envs parallel, engines serve during a block, env steps finish while paused, World episodes on workers |
| 9 | per-phase bubbles | **done** | `idle/rollout_s`, `idle/train_s`, `idle/env_wait_s`, `env/busy_s`, `engine/utilization` per step |
| 10 | colocated_sync on-policy | done | |
| 11 | partial barrier semantics | partial | tested: aborts never replay environment actions; KV release is the engine's sleep (vLLM) |
| 12 | disaggregated never pauses | done | |
| 13 | publish every K, optimizer reset | **done** | K = 2 versions; moments zeroed at publish |
| 14 | warmup batches | done | tested: first step waits for B × 2 terminal groups |
| 15 | per-source admission | done | lower-bound test |
| 16 | terminal-only GRPO, critic no barrier | **done** | critic mode (`TorchTrainer(critic=)`): n = 1 groups are selectable on finish; GRPO at n = 1 has zero advantage (tested), the critic learns (reward 0.04 → 0.21, EV 0 → 0.80) |
| 17 | group backfill | done | pad / drop_group / refill tested |
| 18 | loss_agg + DAPO golden | done | |
| 19 | DP balancing | done | real DP: `DistributedTorchTrainer` |
| 20 | drop ≤ τ, one refill per evict | **done** | tested: one admitted group per evicted group |
| 21 | wait, zero evictions | done | |
| 22 | in-flight ≤ S × B | done | |
| 23 | GLM example | done | |
| 24 | staleness metric set | **done** | `staleness_worst/*`, `evicted_staleness/*` (incl. evictions at blocked barriers — a bug found), `admission_blocked_s` |
| 25 | max_offpolicy_frac | done | |
| 26 | windowed FIFO rule | done | |
| 27 | Fig. 9 reproduction | **done** | W = 1 ≡ strict FIFO, W = N ≡ greedy tested |
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
| 46 | cross-segment GAE | **done** | wired into critic mode; trainer advantages equal direct GAE with segments |
| 47 | skip-observation GAE | done | wired: the critic chain holds completion tokens only |
| 48 | critic knobs | **done** | TTUR (`updates_per_policy_update`), `HFValue(freeze_attention=True)`, λ_policy length-adaptive (α = 1.5), λ_critic = 1, truncation bootstrap; explained variance logged |
| 49 | critical_steps, frozen sub-agents | partial | critical_steps; TorchTrainer frozen mask |
| 50 | prefix sharing | **done** | `TrainerConfig.prefix_sharing`: items with one full prompt in a micro batch encode it once (`TinyPolicy`: GRU state; `HFPolicy`: Forge-style packing, positions restart per completion, 4D *float* mask, since a bool mask is ignored by eager attention); loss and gradients equal the unpacked step within 1e-6; `train/prefix_tokens_saved`. Open: keeping a group in one micro batch when packing |
| 51 | per-step monitor set | **done** | runtime feeds truncation, zero-variance groups, entropy (behaviour estimate), confirmed hacks |
| 52 | pass-rate alarm → re-audit | **done** | `Runtime(on_alarm=)` gets the batch's records; tested with a sudden pass-rate jump |
