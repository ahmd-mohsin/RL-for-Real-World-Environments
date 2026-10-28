# Asynchronous RL audit: Crucible's loop against prime-rl, AReaL, slime, SkyRL, verl, NeMo-RL, TRL, ROLL

**Date** 2026-10-28. **Scope.** Every asynchronous-RL setting and code path that the DeltaAI run uses. The student is
Qwen/Qwen3.6-35B-A3B, run through Crucible's own loop (`bridges/tinker/trainer.TinkerTrainer`) over the prime-rl v0.10.0
bridge (`bridges/primerl/`). The framework-neutral torch loop (`train/runtime.py`, `scheduler.py`, `pool.py`,
`trainer.py`, `weights.py`, `state.py`) is covered too.

**Method.**
- The code was read against the local clones in `RL-Environments/06-frameworks-and-env-code/`. prime-rl was also read at
  v0.10.0 itself: a source copy from the smoke work (the local clone is `0.9.1.dev235`).
- Every bug has a unit test that fails on the code before the fix (checked against a copy of `HEAD`).
- No GPU work and no paid calls.
- Two other workers own adjacent items and are not repeated here: the SC+MIS estimator's math, and the order of step
  metrics in `primerl/backend.py` (`FileIO.metrics`).

Line numbers are those of this commit.

## 1. How the loop is asynchronous today

### 1.1 Rollout to train

- **The bound.** `TrainConfig.max_staleness` (default 0) and `async_groups` (default 0) are set in
  `bridges/tinker/trainer.py:127-128`. `async_groups > 0` requires `max_staleness >= 1` (`:159`).
  - `delta_rl.py` keeps both at 0, so **the planned run is synchronous**: no group samples while a step trains.
  - With `async_groups = a`, a groups more than the batch keep sampling while `_step` awaits the trainer (`:524-525`).
    The launch budget is `max_launches + a` per step.
- **Staleness definition.** A group's staleness is `state.version − min(policy_version of its trainable turns)`
  (`Group.min_version`, `:187`; `_stale`, `:422`). This is prime-rl's definition: the version a batch trains on is
  v{step−1}, minus the oldest version that produced a token (`orchestrator/utils.episode_staleness`).
- **Enforcement: filter only, not admission.**
  - A group is checked when it lands (`_verdict`, `:542`) and again when the batch is built (`:559`).
  - A stale group is **dropped** (`drops.jsonl`), never reweighted or kept.
  - No group was cancelled in flight. prime-rl cancels them (`dispatcher.on_version_pending`); recommendation R3, now
    fixed: a doomed group is cancelled before each weight swap.
- **Versions in flight.** At most one version per call, with no fixed bound per group. A group can span any number of
  versions and is trained only if its oldest version is within the bound.
- **Partial rollouts: call-level switching.**
  - `TinkerSampler.gen` pins the sampler handle and version when a call starts (`bridges/tinker/sampler.py:92`).
  - `refresh` (`:80`) changes them only for later calls. A call in flight finishes on the adapter it started with.
  - The next turn of the same episode uses the new adapter.
  - Each turn records its own `policy_version`, so a multi-turn episode that spans a hot-load carries one version per
    turn.
  - Comparison: AReaL and verl abort the call and resume it under the new weights; prime-rl, TRL and SkyRL swap in place,
    so one call mixes versions. Crucible never mixes versions inside a call.
- **Behaviour logprobs.** Each call's logprobs come from vLLM for the version that sampled them (`parse_completion` /
  `parse_generate`). They are aligned per token by `bridges/common.flatten` (`common.py:85-132`). The trainer's
  `inference_logprobs` are exactly those. Nothing recomputes them or zero-fills them: a token-incomplete call drops the
  rollout (`untrainable`, `common.py:70-82`).

### 1.2 Token-in, token-out

- `EngineLLM` (`train/world_env.py`) extends a conversation only through its own reply nonce. The next prompt is
  `prompt + completion (+ EOS when cut) + the template's suffix`.
- A re-render (thinking strip, a nudge, compaction) opens a new segment, never a stitched sequence.
- `flatten` puts tool output and the appended EOS at mask 0. Only sampled tokens train, including the sampled stop token
  (`cut_at_stop` keeps it).
- Routes and top-k heads are keyed by sha(prompt + completion) of each call (`routing.key_of`). A completion trimmed
  after sampling would miss its key: coverage drops below 1, and that step falls back to no replay or no centering, with a
  warning. It is never a silent mismatch.

### 1.3 MoE routing replay across versions

- `routing.stitch` takes each position from the first call that forwarded it (`routing.py:94-118`).
- Synchronous run: every call of an episode is one version, and the prefix is served from the episode's cache, so replay
  is exact.
- Asynchronous run: a call after a hot-load re-forwards the whole prefix under the new adapter, because the prefix cache
  is keyed by adapter. Its prefix routes then differ from the first call's. The trainer gets one routing per position, so
  the later call's tokens are scored over a prefix routed as the earlier version routed it.
- No framework solves this exactly:
  - slime forks a segment per call when R3 is on (`slime/agent/trajectory.py:200-209`);
  - SkyRL and ROLL concatenate per call and new token.
- See recommendation R5 (fixed by refusal: such a sequence is not replayed).

### 1.4 The prefix cache across versions

- Each version is served under its own adapter name, and vLLM keys prefix blocks by LoRA. KV from one version therefore
  never serves another.
- The generate endpoint's `cache_salt` now names the episode **and** the adapter (`primerl/backend.py:298`). prime-rl
  salts per weight version for the same reason, because it reloads one adapter name in place (`orchestrator/clients.py`,
  `load_lora_adapter` doc).

### 1.5 Temperature and truncation

- vLLM returns processed logprobs (`logprobs_mode = processed_logprobs`, prime-rl's default), and the trainer divides
  logits by each token's `temperatures` entry.
- Before this audit, two things could make them disagree silently. Both are fixed: bugs B1 and B4 below.

### 1.6 Weight sync

- Order for step s:
  1. The bridge sends `batches/step_s`.
  2. The trainer trains and offers `broadcasts/step_s`.
  3. The bridge answers (`.receiver_ready`) and waits for `.finished`.
  4. `/load_lora_adapter crucible-s<s>`.
  5. `save_for_sampler` → `TinkerSampler.refresh`.
  6. `save_state` waits for the DCP `.metadata` of `checkpoints/step_s`.
- Calls in flight keep the previous adapter. New calls use the new one.
- prime-rl's trainer keeps broadcasts s and s−1 (`WeightSender._clean`). A call two versions old whose adapter vLLM has
  evicted could not reload, but calls are at most 8,192 tokens and steps take minutes, so this does not occur.
- Every wait is bounded (`send_timeout_s`, `step_timeout_s`, `startup_timeout_s`; `/load_lora_adapter` is retried).
- **Resume.**
  - Crucible's `state.json` `resume.step` is written only after prime-rl's checkpoint of that step is complete.
  - prime-rl counts steps from 1. Its checkpoint s holds policy v{s} with the optimizer and scheduler. A resumed trainer
    re-offers `broadcasts/step_s` and reads `batches/step_{s+1}` next.
  - The bridge now relaunches the trainer at exactly Crucible's step, with the stale-file hygiene of B2 and B3.

### 1.7 Objective and statistics

- Groups are `group_size` attempts of one task (default 4; the plan uses 8). A group with any masked member is dropped
  whole and refilled (`infra_in_group`).
- Advantages are Dr.GRPO, R − mean R, with no std (`train/credit.outcome_advantages`). Zero-spread groups are dropped and
  refilled (`drop_zero_spread`, DAPO's dynamic sampling). Since the second pass the spread is that of the composed
  advantages (outcome + honesty), so a flat group whose honesty terms differ trains (harness audit R8).
- Loss aggregation: `loss_scale = token_mean` (the default since the second pass, R9) or `sum`. The bridge multiplies by the step's trained-token count
  (`wire.undo_token_mean`, `wire.py:82`) to cancel prime-rl's division by the global rl weight (`train.py:311-327`), so
  both modes train exactly what they train on Tinker.
  - Under AdamW and prime-rl's `max_norm = 1.0`, `sum` and `token_mean` differ only by a per-step scale, which clipping
    and Adam remove.
  - Within a step both weight every token equally, as DAPO's token-mean does.
- The default correction is SC+MIS with no PPO clip (`loss.py`). MIS keeps r in [0.5, 5].
  - Under staleness, MIS is the only trust region. A stale token whose ratio leaves the band is masked, not clipped.
  - `is/masked_frac` against `staleness/max` is the signal to watch (R2: now logged per staleness bucket).
- No reference KL. The sampler anchor β is 0 by default.

### 1.8 Determinism and accounting

- Group seeds are `seed + gi`. Sampled calls pass no seed (`actors/policy.py:360`), so GRPO members differ.
- Resume does not double-train.
  - `next_group` and `cursor` advance on launch.
  - Groups sampled but never trained are lost, not retrained (as in AReaL, ROLL and verl fully-async).
  - The difficulty sampler and the quarantine keep the observations of groups that landed, including stale and dropped
    ones. These are measurements of an older policy, decayed per step.
- With `save_every > 1` (not the default), the steps between checkpoints are retrained after a crash. They appeared
  twice in `steps.jsonl` (R7, fixed: the rows past the resumed step move aside at the restart).

## 2. Bugs found and fixed

Each row has a test that fails before the fix. Tests: `tests/unit/test_async_rl_audit.py` (DeltaAI bridge) and
`test_async_rl_audit_loop.py` (torch loop).

| # | sev | where | what was wrong | fix |
|---|---|---|---|---|
| B1 | **high** | `primerl/backend.py` `_Sampler.sample` / `_generate`; `deploy.inference_toml` | Requests set only `temperature`. vLLM fills omitted fields from the repo's `generation_config.json` (default `--generation-config auto`; Qwen3 repos ship `top_k 20`, `top_p 0.95`), and prime-rl returns processed logprobs. So q could be truncated and renormalized while the trainer scores the full softmax: every ratio and every SC head is biased. prime-rl's own orchestrator always sends `top_p 1.0, top_k −1, min_p 0.0` (`configs/orchestrator.py`, `resolve_env_config`). The smoke's mismatch KL (0.002–0.009) cannot rule it out: a 0.95 nucleus adds only about 0.001. | Found independently by the score-centering audit and fixed there (commit 4717ac39): every call sends `backend.NEUTRAL_SAMPLING`, and vLLM runs with `generation_config = "vllm"`, as `bridges/vllm/serve.py` already does. This branch merged onto that fix and keeps a test of both endpoints. |
| B2 | **high** | `primerl/backend.FileIO`, `PrimeRLBackend.trainer` | The bridge launches the trainer with torchrun, not with prime-rl's `rl` launcher, so prime-rl's resume hygiene (`clean_future_steps`, `entrypoints/rl.py:519`) never ran. The v0.10.0 receiver writes its request and reads `batches/step_<s>/rank_<i>.bin` as soon as the file exists. A job killed after sending batch s+1 but before Crucible recorded step s+1 (likely with 1 h steps under a 12 h limit) left that batch, and the relaunched trainer trained the **stale** batch in place of the new one. The loop then logged the new groups as trained. DCP also rewrites a checkpoint in place with `.metadata` last, so the stale step-(s+1) checkpoint counted before the new one was written. | `FileIO.prepare_resume(s)` runs before the startup handshake, so it cannot race the trainer: it deletes `batches/step_{>s}`, moves `checkpoints/step_{>s}` to `stale/<time>/` (nothing deleted), and refuses a fresh run whose output already holds checkpoints, as prime-rl does. |
| B3 | **high** | `deploy.sbatch_script`, `render --resume-step` | The trainer's resume step was fixed at render time, and nothing tied it to Crucible's `state.json`. A resubmitted job, or one whose loop lagged the trainer by one step, offered a different startup step from the one the loop awaited, and both waited out `startup_timeout_s` (3 h on 4 GPUs). The run needs about 40 h, so it needs several jobs. | The job writes `trainer_run.toml` from Crucible's state at start (`deploy.launch_toml`): the same job file resubmits safely, and a pinned step that disagrees fails before any GPU work. `FileIO.wait_startup` fails at once when a trainer offers another step after the wait began. `_handshake` answers a re-offer when the trainer recreates the directory after the bridge answered a stale offer, the twin of smoke attempt 11's race. |
| B4 | medium | `scripts/delta_rl.py _backend`; `TinkerTrainer.run` | The trainer scores at `PrimeRLBackend.temperature`, which was never wired (always 1.0), while groups sample at `TrainConfig.temperature`. A run at T ≠ 1 (doc 14 §6 suggests a higher-temperature smoke) would have trained with every ratio off. | The wiring in `_backend` is also in 4717ac39. This audit adds the guard: the loop refuses a backend whose training temperature differs from the sampling temperature. |
| B5 | high | `train/scheduler.py select` | The relaxed fill (admission closed, nothing live) re-added groups `_pick` had already chosen, because `_pick` does not record into `taken`. A group was trained twice in one batch (doubled token weight, shifted baselines), and the held-back group was never trained. | `taken` includes the picked groups. A finite source's tail (every source exhausted) trains short instead of blocking until `max_wall_s`. |
| B6 | medium | `train/trainer.py` π_θold guard | The guard refused implicit π_θold only for multi-version rollouts. A single-version rollout at any staleness was recomputed with the current weights (π_θold = π_θ), so PPO, CISPO and GSPO ignored its behaviour logprob: no off-policy correction at all. | The runtime sets `trainer.policy_version` (the served version). Implicit π_θold now also needs every token at that version; otherwise `old_logprob_source` must be explicit. |
| B7 | medium-low | `train/pool.py DataPool.call` | The lazy staleness heap assumed a group's oldest version never falls below its submit version. A call served by an older version (a remote engine still loading after the publish returned) left the bound too high, so `wait` trained at τ+1. | `call` pushes the lower bound when a call's version is below the submit version. |
| B8 | medium | `primerl/routing.RoutingStore` | One entry per sha(prompt + completion). Two calls that return the same tokens collided: group members opening with the same short tool call, or one task in two groups on different adapter versions under async. The second call overwrote the first's routes, so the first sequence replayed another call's (possibly another version's) experts, and training one group dropped the key the other still needed: coverage < 1, so that step replayed nothing. This is the pattern the score-centering audit fixed in `TopKStore` (commit 4717ac39). | A list of calls per key, each kept with its sampled logprobs. `get` / `drop` pick the training sequence's own call by those logprobs (`routing.closest_call` over `masked_logprobs`). `TopKStore` now uses the same helper (`topk.trained_logprobs` is `masked_logprobs`). |
| B9 | medium | `train/trainer.py _adapt_entropy` | mimo4's entropy-adaptive bands were written into `cfg.correction.params` (the user's object). The config is the run identity, so after one step every resume was refused (`TrainResumeMismatch`), and with `allow_change` the bands reset silently. | The bands are trainer state (`TorchTrainer.bands`), saved and restored by `state.capture` / `apply_trainer_state`. The config is never mutated. |

**Checked and left as is.** The in-process and checkpoint publishers advance the version before `load_weights`, so a
failed load uses up its number. This is deliberate: a load that failed part way may already have reached some engines,
so the number must never be reused for other weights. A version directory whose load never confirmed also stays pinned
(`test_m12_audit_scale.py::test_D8_*`).

Also hardened: an asynchronous run gets `vllm.max_loras = 2`. The previous version's calls are in flight when the next
adapter loads, and with prime-rl's single slot vLLM would hold every new call until they drained. TRL's async GRPO names
adapters per version too, and requires `--max-loras >= max_staleness + 2` (`async_grpo_trainer.py:885-903`). Crucible
switches per call, so only one older version is ever needed.

**Adopted from other frameworks** (patterns re-implemented, no code ported, so no THIRD_PARTY rows):
- prime-rl's resume hygiene (`clean_future_steps`) and its neutral train-sampling values;
- prime-rl's per-version prefix-cache salt;
- TRL's per-version adapter names, with enough LoRA slots for them.

## 3. The matrix

Notation: "admission" means a bound on how far generation may run ahead; "filter" means a check on finished groups at
train time. Defaults are the code defaults; example configs are noted where they differ. Paths are relative to
`06-frameworks-and-env-code/`.

### 3.1 Staleness

| framework | bound (default) | where enforced | over the bound | versions in flight |
|---|---|---|---|---|
| **Crucible (DeltaAI)** | `max_staleness` 0, `async_groups` 0 | filter at landing and at batch build | dropped and refilled | per call one; a group any number, trained only within the bound |
| Crucible torch loop | `SchedulerConfig.max_staleness`, strategy `wait` / `drop` | admission (untrained cap) + filter + doomed-group cancel | `drop`: evicted; `wait`: barrier blocks | τ + 1 |
| prime-rl | `max_off_policy_steps` 8 (`configs/orchestrator.py:591`) | in-flight cancel before each swap (`dispatcher.py:402-437`) + queue sweep (`train_sink.py:162-196`) | dropped | ≤ 9 |
| AReaL | `max_head_offpolicyness` 0 (`cli_args.py:2859`; gsm8k example 2) | admission only (`staleness_manager.py:79-113`) | none possible; kept | ofp + 1 |
| slime | none | concurrency / queue only (`fully_async_rollout.py:89-90, 202-207`) | kept; staleness only logged (`utils/staleness.py`) | unbounded |
| SkyRL | `fully_async.max_staleness_steps` 4 (`config.py:1029-1040`) | admission (AReaL-style, `fully_async_trainer.py:181-187`) | kept with a warning and a violation count | ≤ 5 |
| verl one-step-off | fixed 1 (`one_step_off_policy/ray_trainer.py:385-406`) | structural | n/a | 1 |
| verl fully async | `staleness_threshold` 0.1, `trigger_parameter_sync_step` 4 | admission (`fully_async_rollouter.py:1142-1160`) | oldest evicted when the queue is full | ≈ threshold × window |
| verl v1 trainers | `max_off_policy_threshold` 8, strategy `drop` (`ppo_trainer.yaml:298-303`) | sample time (`replay_buffer.py:520-613`) | dropped and refilled, or `wait` | ≤ 8 |
| NeMo-RL | `max_trajectory_age_steps` 1 (`grpo.py:273-287`) | admission (target version) + filter (`replay_buffer.py:540-673`) | evicted | age |
| TRL async GRPO | `max_staleness` 4 (`async_grpo_config.py:347-379`) | filter at train time (`async_grpo_trainer.py:362-367`) | dropped | ≤ 5 |
| ROLL | `async_generation_ratio` 0 (`base_config.py:552-557`) | buffer selection + gc abort (`generate_scheduler.py:358-442`) | aborted / dropped | ⌈ratio⌉ + 1 |
| OpenRLHF | `async_queue_size` 1 (`train_ppo_ray.py:294`) | admission (semaphore) | n/a | 2 |

### 3.2 Calls in flight at a weight update, version tags, prefix cache

| framework | in-flight calls at an update | version tag | prefix cache across versions |
|---|---|---|---|
| **Crucible (DeltaAI)** | finish on their own adapter; the next call switches | per call (`Turn.policy_version`; `version_spans` if a call ever crosses) | separate adapter names + salt (episode, adapter) |
| prime-rl | in-place LoRA reload under live traffic: one call can mix versions | per episode span (`policy.start` / `end`) | salted per weight version |
| AReaL | abort, then resume with prompt + tokens under the new version (`remote_inf_engine.py:1105-1198`) | per token (`output_versions`) | prefix re-prefilled |
| slime | abort and requeue (default `flush_cache_interval` 1), or pause in place | per segment (`weight_versions`) | flushed on abort |
| SkyRL | keep-pause, then continue in place; old KV reused unless cleared | per group (`global_step_when_scheduled`) | `use_cache_salt` per weight version |
| verl fully async | abort, then resubmit prompt + tokens (`llm_server.py:290-361`) | per trajectory (`min/max_global_steps`) | reset on abort |
| NeMo-RL | drain by default; `in_flight_weight_updates` keep-pause | per batch (`generation_weight_version`) | optional recompute |
| TRL async GRPO | `/pause?mode=keep`, continue on new weights | per group (`model_version`) | adapter `{lora}-v{n}`; `max_loras ≥ staleness + 2` |
| ROLL | abort and resend (RLVR) / regenerate the turn (agentic) | per prompt (`sampling_start_step`) | n/a |
| OpenRLHF | lock (no overlap) or keep-pause | none | reset if enabled |

Crucible's call-level switching is the only scheme in the table where no single call mixes two versions, while
episodes still move to new weights between calls. It suits 300-call episodes: a call is at most 8,192 tokens, and an
episode is about 28 minutes.

### 3.3 Behaviour policy, correction, objective

| framework | π_old in the ratio | train/inference correction (default) | loss aggregation | advantage | zero-variance groups | KL |
|---|---|---|---|---|---|---|
| **Crucible (DeltaAI)** | sampler logprobs of each call | SC + MIS [0.5, 5], k = 32; no PPO clip | `token_mean` (was `sum`; both exact via `undo_token_mean`) | Dr.GRPO + honesty + M6 turn credit | dropped and refilled when the composed advantages are equal | none (anchor β 0) |
| prime-rl | sampler | custom / IcePop-style masks | global rl weight | per env | filtered (`no_signal`) | optional |
| AReaL | sampler by default; decoupled: recomputed prox + behaviour IS (`functional.py:452-596`) | rejection sampling off by default (`behav_imp_weight_cap` removed) | global token mean | none by default | user filter | `kl_ctl` 0.1 vs ref |
| slime | trainer recompute (default) | `--use-tis` off; tis 2.0 / mis 0.5–5 | per-sample token mean | GRPO with std | `check_reward_nonzero_std` | 0 |
| SkyRL | trainer recompute | off-policy correction off by default | token mean | GRPO with std | `zero_variance_filter` off | k3 0.001 |
| verl | rollout logprobs (bypass) in async recipes | `rollout_is` off; `score_centering` option (`rollout_correction.yaml:32`) | token mean | GAE / GRPO with std | none by default | off |
| NeMo-RL | trainer recompute; IS required for async | TIS / IcePop optional | token mean | leave-one-out, normalized | 0 advantage | k3 0.01 |
| TRL async GRPO | sampler logprobs | none in the local clone (doc 14 §7.6 cites SC+MIS at main 9824938; the local clone, 1.16.0.dev0, has no `score_centering`) | token mean | group mean / std | kept (0) | none |
| ROLL | recompute (forced in most configs) | `train_infer_correction` off | seq-mean-token-mean | GAE default | `no_filter` | 0.2 if enabled |

### 3.4 Resume

| framework | saved | in-flight groups on resume | step counter |
|---|---|---|---|
| **Crucible (DeltaAI)** | `state.json` (step, sampler path, quarantine, difficulty, cursor) after prime-rl's checkpoint is complete; prime-rl weights + optimizer + scheduler | lost, never trained twice; trainer relaunched at Crucible's step with future batches and checkpoints cleared | prime-rl from 1; Crucible version v = prime-rl step v |
| prime-rl | trainer checkpoint; orchestrator checkpoint | `clean_future_steps` deletes future batches and broadcasts | from 1 |
| AReaL | `RecoverInfo` with the dataloader | skipped | version = step + 1 |
| slime | data source with buffer and ready queues | kept (buffer saved) | `rollout_id + 1` |
| SkyRL | consumed / filtered uids | regenerated | `global_step` = step in progress |
| verl fully async | dataloader `data.pt` | lost (warned) | 0 / 1 |
| NeMo-RL | replay buffer + dataloader | restored, with old entries dropped | 0 |
| TRL async GRPO | `rollout_state.json` (prompt index, version) | regenerated | HF `global_step` |
| ROLL | dataset cursor | skipped | `state.step` |

## 4. Recommendations

A second pass (2026-10-28, branch `worktree-agent-a6734480fd8877554`) implemented or decided R2-R9. Each fix has a test
that fails on the code before it: `tests/unit/test_open_risks_loop.py` (R2, R3, R4, R6, R7) and
`test_open_risks_routing.py` (R5). R1 is a configuration choice for after the GPU run.

| Item | Status |
|---|---|
| R1 overlap sampling and training | open: a configuration choice after the 2-step real-model run |
| R2 trust region per staleness bucket | fixed (all buckets' token shares; the KL and MIS share where the backend returns trainer logprobs) |
| R3 cancel doomed groups in flight | fixed |
| R4 cancel pending groups at an error stop | fixed (at every stop) |
| R5 version-consistent routing replay | fixed by refusal (a sequence that spans a weight swap is never replayed) |
| R6 the quarantine under async | fixed |
| R7 `save_every > 1` duplicates rows | fixed |
| R8 `publish_every > 1` in the torch loop | decided: documented, not a bug |
| R9 Dr.GRPO's constant normaliser | decided: `token_mean` is the default; a constant normaliser is not adopted |

- **R1. Overlap sampling and training before going asynchronous in earnest** (`--async-groups 4 --max-staleness 1`).
  Open, by design.
  - Sampling dominates (doc 14 §3: about 1 h per 4 × 8 step, of which training is about 0.2 h). In synchronous mode the
    vLLM GPUs idle while the trainer runs, and the trainer GPUs idle for the other ~0.8 h.
  - With call-level switching, staleness 1 costs only the calls in flight at the swap, and MIS bounds them.
  - Run it after the 2-step real-model run has shown `kl/mismatch`, `sc/*` and `is/masked_frac` at staleness 0.
  - prime-rl defaults to 8, SkyRL and TRL to 4, NeMo-RL to 1, and AReaL's example uses 2.
- **R2. Watch the trust region under staleness.** **Fixed.**
  - Each step logs, per staleness bucket b (the version the step trains on minus a sequence's oldest version), the share
    of trained tokens: `staleness/tokens_frac/s<b>` (`trainer.staleness_metrics`).
  - Where the backend returns the trainer's logprobs (Tinker), it also logs `kl/mismatch/s<b>` = E[r − 1 − log r] and
    `is/masked_frac/s<b>`, the share of tokens whose ratio is outside MIS's [0.5, 5].
  - prime-rl's trainer returns no per-token logprobs and its loss sees no per-sequence tag, so on the DeltaAI path only
    the token shares are logged. In a step with one bucket, which is every synchronous step, the trainer's own
    `is/masked_frac` and `kl/mismatch` are that bucket's. A per-bucket split there needs a per-sequence channel into
    prime-rl's `compute_loss` (the score-centering hook's pattern); it is not built, because the planned run is
    synchronous.
  - Found on the way: `staleness/max` and `staleness/mean` were measured against the version after the step, so a
    synchronous run logged 1. They now use the version the step trains on (0 when synchronous), the definition the
    filter uses.
  - If the masked share grows with staleness, add AReaL's decoupled proximal policy (a recompute pass per step, which
    costs one more forward pass at 160K tokens), or lower `max_staleness`.
- **R3. Cancel doomed groups in flight.** **Fixed.**
  - `Runner.inflight_min_version(run_id)` reads the oldest policy version among the calls of the attempts still running
    (`Runner.live`, `Attempt.policies`, set by the harnesses; a discarded summary attempt is left out). Every later call
    samples a newer version, so the value can only stay or fall.
  - At the start of each step (`TinkerTrainer._cancel_doomed`, before the weights move to v{step}), a group still in
    flight whose oldest token is more than `max_staleness` behind the new version is cancelled. It is a
    `cancelled_stale` drop with its minimum version, and it counts as lost for the quarantine.
  - This is prime-rl's `on_version_pending` and the torch loop's `Scheduler.evict_doomed`. Only asynchronous runs have
    groups in flight at a step. A 300-call episode holds its Worlds for about 28 minutes, which used to run out before
    the group was dropped as `stale` on landing.
- **R4. Cancel pending groups on an error stop.** **Fixed, at every stop.**
  - `TinkerTrainer.run` no longer awaits the groups in flight in its `finally` (up to `rollout_s`, 30 minutes, holding
    the GPUs). It cancels them (`_stop_pending`), waits at most `cancel_grace_s` (300 s) for their Worlds to close, and
    drops them as `cancelled_at_stop:<why>` (`error:<type>` on an exception). A group that had already finished is
    `unused_at_stop:<why>`, as before.
  - Cancelling at a normal stop too is deliberate: a group in flight at a stop can never train, and its observations
    for the quarantine and the difficulty sampler are lost either way (as after a crash).
- **R5. Version-consistent routing under async.** **Fixed, by refusal.**
  - A sequence whose sampled tokens span more than one policy version (`meta` min_version ≠ max_version) is never
    replayed. `wire._routes` sends no routes for it (coverage 0, `routing_version_mixed`), so `replay_all_or_none`
    recomputes the router for that step, with one warning per step and `routing/version_mixed_share` on the dashboard.
  - Before, `stitch`'s first-forward-wins rule trained the later version's tokens over a prefix routed as the earlier
    version routed it.
  - The synchronous DeltaAI run has one version per episode and always replays. Under async, a step with an episode
    that crossed a swap trains without replay.
  - The exact alternative is to split the sequence at the version change (slime's per-call fork), which forwards the
    prefix again. It is not built: measure the refused share first.
- **R6. The quarantine under async.** **Fixed.**
  - The first group to land after a probation opened decided the release, and it could be a group picked before the
    quarantine. A late group could also strike, or re-quarantine, a task already quarantined.
  - Each group now carries the step its task was picked at and whether it was the probation retry
    (`TaskQuarantine.last_probe` after the pick). `observe(..., picked=, probe=)` ignores a group picked before the
    task's last quarantine (`_Task.since`), and during a probation only the retry group decides. Strikes are dated by
    the pick step. `lost(..., probe=)` re-opens a probation only when the retry itself was lost.
  - Callers that pass neither keep the old behaviour. The difficulty sampler forwards both and marks its probation pick
    the same way.
- **R7. `save_every > 1` duplicated rows.** **Fixed.**
  - At a restart, the rows of `steps.jsonl` and `rewards.jsonl` past the resumed step (steps trained after the last
    `save_state`, which the resume trains again) move to `steps.rolled_back.jsonl` / `rewards.rolled_back.jsonl`
    (`TinkerTrainer._rollback_logs`). Nothing is deleted. A torn last line moves too.
  - A dashboard sink that already received those steps (W&B) still has them; the files are the record.
  - `save_every = 1` stays the default, and prime-rl checkpoints every step anyway.
- **R8. `publish_every > 1` in the torch loop.** **Decided: not a bug; documented.**
  - τ counts published versions, by definition (`SchedulerConfig.max_staleness`: "versions behind"). With
    `publish_every = k`, a token sampled from version v may be up to τ·k + (k − 1) optimizer updates behind the weights
    that train it.
  - The DeltaAI loop publishes every step (`TinkerTrainer`), so it is unaffected. A torch-loop run that publishes less
    often should set τ for the update gap it accepts (τ ≈ ⌊gap / k⌋). DESIGN §13 says so.
- **R9. Dr.GRPO's normaliser.** **Decided** (with the harness audit's R9).
  - `loss_scale = token_mean` is now the default: a global token mean, as prime-rl, AReaL, SkyRL, verl, NeMo-RL and TRL
    normalise (§3.3), and on the prime-rl path exactly prime-rl's own normalisation. Within a step every token weighs
    the same (DAPO's token mean, Dr.GRPO's unbiased choice); across steps the scale no longer swings with the batch's
    token count. With `sum`, the clip (`max_norm` 1) bound on almost every step and gave every step the same length
    whatever its signal.
  - A constant normaliser (the generation budget) would also keep the step size steady. Under AdamW its only gain over
    `token_mean` is when clipping stops binding, so it is not adopted.
  - A run directory made with the old default (`sum`) resumes only with `--loss-scale sum`.
- **Test isolation (fixed).** `tests/unit/test_primerl_score_centering.py::test_install_patches...` failed only in
  full-suite order. The cause was production code, not a test. `sc_head.install` found prime-rl's train module by
  `getattr` over all of `sys.modules`. A lazy module that any process importing transformers holds (an aria
  image-processing submodule) imported torchvision on attribute access and raised `ModuleNotFoundError`. So the test
  failed after any test that imported transformers (`train/bridges/tinker/codec.py` does), and the trainer's hook
  depended on what else its process had imported. The scan now reads each module's own `__dict__`. Regression test:
  `test_open_risks_routing.py::test_sc_head_install_is_not_broken_by_a_lazy_module_in_sys_modules`.
