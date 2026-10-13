# M12 framework sweep, group 3: slime, AReaL, ROLL (async-native runtimes)

Read-only sweep of M12 `train` at Crucible HEAD `3b522cb` (2026-10-12). I opened every Crucible and external citation myself. Nothing under `crucible/src` or `crucible/tests` was changed.

Path roots:
- `AR` = `RL-Environments/06-frameworks-and-env-code/infra-inclusionai-areal/`
- `SL` = `RL-Environments/06-frameworks-and-env-code/infra-thudm-slime/`
- `RO` = `RL-Environments/06-frameworks-and-env-code/infra-alibaba-roll/`
- `C` = `crucible/src/crucible/train/`; `T` = `crucible/tests/unit/`

Probes `p1`..`p13` live in `scratchpad/m12/g3/`. They run from `crucible/` with `PYTHONPATH=src .venv/bin/python` (torch 2.14, CPU). AReaL code is loaded from source with stubbed imports: `p1` loads `functional.py` and `p5` loads `staleness_manager.py`. `[ran]` means a probe reproduced the finding. `[read]` means I found it by reading code only.

Verdicts:
- **DEPEND**: use their code or contract as is.
- **PORT**: copy and adapt.
- **PATTERN**: reimplement the idea.
- **nothing**: Crucible is as good or better.

## Pins and licences

| Root | Commit | Licence | Installed |
|---|---|---|---|
| `AR` | `298412a` | Apache-2.0 (`AR/LICENSE`, SPDX headers) | no; pure-python files loaded with stubs |
| `SL` | `c933787` | Apache-2.0 (`SL/LICENSE`) | no; read only |
| `RO` | `581046a` | Apache-2.0 (`RO/LICENSE`) | no; read only |

---

## 0. Headline defects

| # | Sev | Defect | Crucible | Evidence |
|---|---|---|---|---|
| D1 | **P0** | **`max_offpolicy_frac` silently breaks the staleness bound.** The frac filter (C/scheduler.py:205-210) removes off-policy groups from `cands` *before* the at-bound rule (`:223`), which only looks at `cands[B:]`. Filtered groups are neither trained nor evicted. They keep aging, and later train far past τ. `wait` checks live groups only (`:167`), so nothing stops it. No monitor alarm fires. | C/scheduler.py:205-223 | [ran] `p8` real Runtime, colocated_partial, τ=1, S=3, frac 0.25: `staleness/max` goes 0,1,…,8 and stays at 8 for 30 steps; `monitor.alarms == []`. Without frac the max is 1. `p4` sim, τ=2, frac 0.25, wait: max 4. `p3` direct scheduler, τ=2, frac 0.5: per-step max 0,1,2,3,4,5,6,6. Repo config of T/test_train_sched.py:105 (τ=3, frac 0.25): max staleness **8**, and the test passes (D4). |
| D2 | P1 | **`max_offpolicy_frac` deadlocks `disaggregated_stream`.** After the first publish every buffered group is off-policy. Only `floor(frac·B)` < B of them are selectable. They fill the S·B·n cap, so no on-policy group is ever admitted. | C/scheduler.py:205-221 | [ran] `p4` sim n=4, B=4, τ=2: steps **1/30** for frac ∈ {0, 0.25, 0.5} under both wait and drop (the event queue empties at t=1026). |
| D3 | P1 | **`windowed_fifo` deadlocks.** The window is `submit_index < head + W` over submit-index *slots* (C/scheduler.py:196-198). Groups trained out of order still occupy their slots until the head passes. Once fewer than B terminal groups fit in `[head, head+W)` while the cap is full, nothing trains or admits again. W = ⌈0.3·S·B⌉ is also < B for every S ≤ 3. | C/scheduler.py:72-73, 195-199 | [ran] `p6`/grid, sim n=4, default frac 0.3: τ=1 → 0/10 steps (B = 2, 4, 8); τ=2, B=4 → 0/10; τ=2, B=8 → 0/10; τ=4 → 1-2/10 (W=5 ≥ B=4: groups 2,5,6 in window, 3,4 trained holes, 16 terminal groups, cap 64/64). [ran] `p7` real Runtime, τ=2, B=4: **0/12 steps, `stop_reason=wall_clock` after 15 s**, untrained=16, live=0. |
| D4 | P1 | **Two vacuous scheduler tests back "done" checklist items #25 and #27/#28.** (a) T/test_train_sched.py:80-82 asserts `max_staleness() <= 10` on a sim that trains **0 of 12 steps** (W=4 < B=8, D3), so `max([])=0` passes. M12-train.md §4 says "the article's lag bound of 10 at batch 8, W 4 is tested", which is false. (b) `:105-106`: the bound `max(1, len(b)//4)*4` equals the batch size (16 rollouts), so it can never fail, and the run it checks trains at staleness 8 with τ=3 (D1). | T/test_train_sched.py:80-82, 105-106 | [ran] config of `:80`: `steps 0/12, blocked {'need_more': 1}`. Config of `:105`: steps 15, max staleness 8, assertion bound [16, 16, 16] for batches of 16. |
| D5 | P1 | **`refill_member` has no refill cap, so one flaky task livelocks or kills the run.** Each select re-refills masked members (C/scheduler.py:146-150). `Group.refills` is counted (C/pool.py:168) but never bounded. Under `wait` the open group's finished member ages ≥ τ and blocks every batch (C/scheduler.py:167-171). | C/scheduler.py:146-150, C/pool.py:167-172 | [ran] `p8`: one prompt whose first member succeeds and whose later members raise `EnvInfraError`. Breaker off: **19,692 refills, 2/30 steps in 20 s, `wall_clock`**. `p9`, default breaker: the whole run stops at **step 1** with `masked_share` (9 masked rollouts, all from that one task). With `pad`: 30/30 steps. |
| D6 | P1 | **A resumed call aborted before its first new token gets a duplicate span start, so the rollout is masked (lost).** Engines append `(len(resume), version)` when a resume starts, not when a token is produced. A second abort with 0 new tokens followed by a resume on v+2 gives `[(0,v0),(k,v1),(k,v2)]`. `Turn.spans_valid()` requires strictly increasing starts (core/schema/rollout.py:68-77), so the call is `missing=['version_spans']` and M10 masks the rollout as `capture_incomplete`. This happens with SGLang (abort while queued or during re-prefill of a long context) and with back-to-back budget barriers. AReaL stamps a version per *emitted token* (AR/areal/infra/remote_inf_engine.py:1165-1167), so it cannot occur there. | C/sglang_http.py:84-86, C/engine.py:355-357, C/hf_engine.py:67-69 | [ran] `p11` drives `SGLangServerEngine.generate` with a stubbed server (abort after 4 tokens on v0, abort with 0 tokens on v1, stop on v2) plus the runtime's resume code (C/runtime.py:266-271): `version_spans=[(0,0),(4,1),(4,2)] spans_valid=False`. `p10`: TorchEngine cannot reach it (no await before the first token). |
| D7 | P1 | **No non-finite grad-norm guard.** `clip_grad_norm_` then `opt.step()` runs unconditionally (C/trainer.py:650-652), so one NaN gradient makes every parameter NaN while `train/loss` stays finite. AReaL skips the step (AR/areal/engine/fsdp_engine.py:742-744) and so does slime (SL/slime/backends/megatron_utils/model.py:668-680). The trigger inside Crucible is the gate pattern `torch.where(gate, r·Â·logp, 0)` (C/mismatch.py:122, 133, 140): when `r = exp(logp − rollout_lp)` overflows, the unselected branch's gradient is `0·inf = NaN`. Behaviour logprobs are pydantic-finite (core/schema/base.py:28), so the triggers are gaps above 709 nats in float64 (e.g. a −1e4 sentinel) or above 88 nats on MPS, where the loss dtype is float32 (C/trainer.py:673-675). | C/trainer.py:650-652, C/mismatch.py:122,133,140 | [ran] `p2`: one token with `logprobs=-1e4` through `TorchTrainer.step` with dis/icepop/tis gives `grad_norm=nan` and **704/704 params NaN**; the `-90` and `-200` cases stay finite (float64). [ran] `p1` (float32 tensors): dis, icepop and mimo4 each give 1 NaN gradient entry with a finite loss for behaviour lp ∈ {−inf, −120, NaN}. AReaL's loss on the same inputs has 0 NaN entries, because it sanitises the log-ratio (AR/areal/utils/functional/functional.py:229-231) and zeroes the *detached* weight (`:420`). |
| D8 | P2 | `VLLMServerEngine.generate` never sends `resume_completion` (C/vllm_http.py:67-71). `version_switch="token"` silently degrades to call-level (the partial is discarded), while `resumed_calls`, `kv_resumes` and `reprefill_tokens` are still counted as token resumes (C/runtime.py:266-275). `capabilities()` does not declare resume support. | C/vllm_http.py:60-71 | [read] |
| D9 | P2 | Subgroup weight updates create a new process group on **every** call (`dist.new_group`, C/weightsync.py:227) and never destroy it. On NCCL each one is a fresh communicator (seconds plus GPU buffers). This is the cost the zhihu article criticises; slime builds its update group once (`connect_rollout_engines_from_distributed`, SL/.../update_weight_from_distributed.py:270-323) and destroys it only on reconnect. | C/weightsync.py:225-229 | [ran] `p12` gloo world 1: process groups 1 → **11** after 10 `update(ranks=[0])`. |
| D10 | P2 | Barrier order is abort-then-pause (C/runtime.py:708-709). With engines whose `abort_all` and `pause` yield (every HTTP engine), aborted calls are re-issued on the old version in the gap. SGLang/vLLM `pause` aborts them again (`/pause_generation` default mode, `/pause?mode=abort`), but `HTTPEngine` → `engine_app` `/pause` → `TorchEngine.pause` does not abort, so those calls run during training. slime and AReaL close admission first: client `is_paused()` (AR/.../remote_inf_engine.py:1114), then pause or abort on the server (`SL/.../update_weight_from_distributed.py:113-120`). | C/runtime.py:708-709 | [ran] `p7`, colocated_partial, 12 steps. Yielding engine: **26** aborted calls vs 8 for the in-process engine, and up to 8 sequences alive at train start. Version labels stay correct because weights are pinned per call. |
| D11 | P2 | Version labels are client-side counters. `SGLangServerEngine.load_weights` sets `self.version` only after the `/flush_cache` await (C/sglang_http.py:130-134). In disaggregated mode a call starting in that window is served by v+1 but labelled v (over-counts staleness; logprobs stay correct). SGLang returns `meta_info.weight_version` and slime records it per segment (SL/slime/utils/types.py:487-488). Crucible ignores it. | C/sglang_http.py:63, 126-135 | [read] |
| D12 | P2 | KPop differs from AReaL. AReaL multiplies the gated loss by `π_prox/π_behave` for every metric including `binary_kl` (AR/.../functional.py:565-568). Crucible's `kpop` gates only (C/mismatch.py:123-128). Pick one and cite Ling 2.6. | C/mismatch.py:123-128 | [ran] `p1`: Δloss 2.2e-3, max\|Δgrad\| 1.6e-3. Every other correction matches exactly (below). |

Non-defects that a reader might expect, checked:
- [ran] `p6` (17 adversarial sim schedules) found zero double-training (`dup_trained=0` in every case) and full rollout conservation (tag statuses sum to admitted in every case).
- Per-source τ holds (a ≤ 1, b ≤ 4).
- `bound_overflow=allow` reaches τ+1 only, as documented.
- An all-masked source stops with 0 steps (the breaker covers this in the runtime).
- Retention `max_cuts=0` on an all-long workload still trains 25/25 steps (96 groups evicted).

---

## 1. AReaL (`AR`, 298412a)

### 1.1 Admission and staleness

| Aspect | AReaL | Crucible | Verdict |
|---|---|---|---|
| Capacity | `min(max_conc − running, (η + v + 1)·B − (accepted + running))`; `accepted` is cumulative and never decremented on consumption (AR/areal/infra/staleness_manager.py:99-113, 150-157) | untrained (live + buffered) ≤ S·B·n, with S = τ (C/scheduler.py:68-70, 110) | Equivalent shape. **Off by one:** AReaL keeps (η+1) batches in flight, Crucible keeps τ. Crucible τ=1 is near-synchronous; AReaL η=1 is one batch ahead. Document the mapping (η ≡ S−1). |
| Version used | engine version at **submit** (`version_provider.get_version()`) | `pool.submit_group(..., version)` at admit (C/runtime.py:538); staleness = v_train − min(call versions), falling back to the submit version (C/pool.py:47-55) | nothing |
| Consumption bound | none. No consumption-side check; only logged (`max_head_offpolicyness` bounds the *count*, `:92-97`) | `wait` blocks while any live group's oldest version ≥ τ (C/scheduler.py:167-171); `drop` evicts terminal groups > τ; at-bound groups train first | **Crucible beats.** [ran] `p5` below |
| Rejected / filtered | `on_rollout_rejected` frees the slot without counting as accepted (`:159-166`); a workflow exception is also a rejection (AR/.../workflow_executor.py:1462-1489) | evicted groups leave the untrained index at once (C/pool.py:113-115) and `admit` refills | equivalent |
| Batch order | oldest **creation time** among finished results, then shuffled (AR/.../workflow_executor.py:316-334; `create_time` = submit, async_task_runner.py:41-42, 444) | `oldest_first` by submit index (C/scheduler.py:191-192) | equivalent |
| Resume after checkpoint | `on_version_recovered` resets `accepted = v·B` (`:115-131`) | the pool is restored from the checkpoint (C/state.py) | nothing |

[ran] `p5`: AReaL's own `StalenessManager` in a DES of its loop (infinite engine, group = slowest of n=4, train 20 s, publish 2 s, B=4, 40 steps), against Crucible `sim.simulate` disaggregated on the **same** `default_shapes`.

| Workload | η/τ | AReaL max head staleness | AReaL share > η | AReaL wall | Crucible `wait` (max / wall) | Crucible `drop` (max / wall / evicted) |
|---|---|---|---|---|---|---|
| tail 0.2×10 | 2 | **6** | 37.5% | 3,614 | 2 / 8,098 | 2 / 7,244 / 37 |
| tail 0.2×10 | 4 | **12** | 36.2% | 2,089 | 4 / 4,541 | 4 / 3,288 / 56 |
| tail 0.05×60 | 2 | **24** | 12.5% | 5,119 | 2 / 24,848 | 2 / 13,161 / 22 |
| tail 0.05×60 | 4 | **35** | 10.0% | 2,709 | 4 / 18,297 | 4 / 4,072 / 21 |

With 50% of groups rejected by `should_accept_fn` at η=2, AReaL's max head staleness is 4 and 13.1% of samples exceed η.

Reading:
- AReaL bounds throughput, not staleness. Its long tail trains up to 35 versions stale and relies on the decoupled loss to absorb it.
- Crucible holds τ exactly. `wait` pays 2-9× wall time on extreme tails (trainer idle 82-97% with an infinite engine).
- `drop` sits within 1.5× of AReaL at τ=4 and still holds the bound.
- If AReaL's throughput is wanted, Crucible already has `staleness_strategy="none"` with S as admission. Name it in the doc as the AReaL-equivalent mode.

### 1.2 Partial rollout and interruption

- AReaL interrupts *inside* `agenerate`:
  - the client loop waits `while is_paused()` (AR/.../remote_inf_engine.py:1105-1115);
  - it pins `request_version = get_version()` per segment (`:1117-1121`);
  - it re-sends `input_ids += output_tokens` (`:1175-1182`);
  - it stamps versions **per emitted token** (`:1165-1167`).
  - The workflow task is never resubmitted.
- **The zhihu digest (07-articles/zhihu-…:46-49) is stale here.** At 298412a, `appendleft` onto `_pending_inputs` happens only on `TaskQueueFullError`, before the task ran (AR/.../workflow_executor.py:451-461). Its §10 "discarded if version diff > staleness" is also not AReaL: there is no consumption-side drop. ROLL drops by `create_step` (RO/roll/distributed/scheduler/rollout_scheduler.py:537-541).
- Crucible `version_switch="token"` is the same mechanism (C/runtime.py:264-275). It records per-call spans instead of per-token versions. The span representation is cheaper but has the zero-token edge in D6. Fix: append a span only when the segment emits ≥ 1 token, or drop empty spans at `finish` (PORT AReaL's per-token rule, S).
- AReaL's SGLang pause is two-stage: abort, then `mode=in_place` for memory release (research 02 §1.3, `sglang_remote.py:371-391`). Crucible's SGLang `pause` is one call with the default mode followed by `release_memory_occupation` (C/sglang_http.py:100-106). It works because the default mode aborts.

### 1.3 Decoupled PPO (three policies)

- AReaL's three policies:
  - π_behave = engine logprobs;
  - π_prox = a forward pass at the step-start weights;
  - π_θ.
- AReaL's loss: `ratio = exp(logp − prox)`, clipped; `behave_imp_weight = exp(prox − behave)` from `apply_rejection_sampling`, multiplied in; normalised by the pre-rejection token count (AR/.../functional.py:452-597, 185-437; AR/areal/trainer/ppo/actor.py:1123-1196).
- **Crucible supports it exactly.** Use a `NEEDS_OLD` correction with `old_logprob_source=None` (single-version rollouts) or `"recompute_current"`. In both cases `TorchTrainer.step` computes `old` once with the pre-update weights, before the update loop (C/trainer.py:610-619). That is π_prox, not π_θold, for multi-version rollouts. `r_dis = π_prox/π_rollout` is AReaL's behave weight and conflates drift with engine mismatch, as AReaL's own `stats.py:19-22` says. The mismatch.py docstring line 28 ("recompute_current = π_θ detached, valid when K = 1") describes only the fallback path inside `correct()` (C/mismatch.py:95-96). The trainer never takes that path. Reword it.

[ran] `p1`: AReaL `ppo_actor_loss_fn` against `crucible.train.mismatch.correct`, with π_prox and π_behave given and B=4, T=32 (two partially masked rows).

| AReaL config | Crucible | \|Δloss\| | max\|Δgrad\| |
|---|---|---|---|
| RS mask, ratio ≤ 5 (AReaL default action) | `icepop(alpha=0, beta=5)` | 0 | 0 |
| RS mask [0.5, 5] (production MoE config) | `icepop(0.5, 5)` | 0 | 0 |
| RS clamp [0, 2] (slime/verl TIS) | `tis(C=2)` | 0 | 0 |
| no RS | `ppo` | 0 | 0 |
| `c_clip=3` | `ppo(dual_clip=3)` | 0 | 0 |
| `binary_kl` ≤ 0.01 | `kpop(phi=0.01)` | 2.2e-3 | 1.6e-3 (D12) |

Gaps against AReaL's loss family:
- `prox_logp_method=loglinear`, which interpolates π_prox from versions and skips the extra forward (`actor.py:1371-1452`; 27% faster per AR docs). PORT, M.
- `reuse_train_logp`. Crucible's fallback path is the same thing when `updates_per_batch=1`. nothing.
- Sequence-level RS (`level=sequence`, geometric mean, `agg` max/mean). Crucible's `mispo`/`hispo` cover the sequence gate. nothing.
- The non-finite sanitisation of D7. PORT, S.

### 1.4 Weight update, engine versions, interface, proxy

| Aspect | AReaL | Crucible | Gap |
|---|---|---|---|
| Path | `xccl`: per-param `full_tensor()` (collective) → rank 0 casts to the compute dtype (bf16) → 1 GiB buckets → single-pending-bucket async broadcast on a side stream → SGLang/vLLM native `update_weights_from_distributed` (AR/areal/engine/fsdp_engine.py:1682-1772, 1442-1528). `disk` path also exists | `DistributedPublisher`: `trainer.state_dict()` (FSDP: full fp32 model to rank-0 **CPU**, C/fsdp.py:163-171) → bucketed double-buffered broadcast → `EngineReceiver` (in-process engines only) (C/weightsync.py:296-365) | Crucible materialises the whole fp32 model on one host per publish and ships fp32 (2× AReaL's bytes). Its receivers cannot be SGLang or vLLM servers; those go through `CheckpointPublisher` (disk). P2 |
| Receiver memory | in-place load by the engine | `StagedState` clones every tensor, then commits (C/weightsync.py:150-161): a 2× model copy on the engine | atomicity costs 2× memory; infeasible for a 70B engine on H100 |
| Pause order | `pause_generation()` before the first bucket, `continue_generation()` after (`:1695-1698`, `:1765-1768`) | colocated: abort, pause, train, publish, resume (C/runtime.py:708-745); disaggregated: no pause, commit at a call boundary | D10 |
| Version on the engine | `set_version` after the update; per-segment pin on the client | per-call pin (C/sglang_http.py:63, C/engine.py:351) | same; D11 |
| Workflow API | `RolloutWorkflow.arun_episode(engine, data)` with per-sample env init in user code (`workflow_executor.py:1337-1341`) | `rollout_fn(rid, prompt, seed, gen, ...)` per source (C/runtime.py:186-189) | nothing |
| OpenAI proxy / TITO | `InteractionCache` parent/child tree by message-prefix match. The child prompt is parent input + output (no stop) + the re-templated suffix aligned on the k-th EOS (AR/areal/experimental/openai/client.py:694-768). Rewards and discount live on the cache (`cache.py:89-178`) | `chat.py` exact TITO (prompt + sampled completion + EOS + rendered suffix). M12-train.md §8 lists "AReaL proxy bridge" as remaining | PATTERN (M): an OpenAI and Anthropic endpoint front for black-box agents, recording into `Turn`s |
| Failure classes | proxy: `context_overflow` → trained 0, `system_error` → reject (`proxy/workflow.py:284-323`) | M10 blame/mask taxonomy | nothing |

---

## 2. slime (`SL`, c933787)

| Aspect | slime | Crucible | Verdict |
|---|---|---|---|
| Rollout plug-in | `--rollout-function-path`; `generate_rollout_async` runs until `rollout_batch_size` groups are kept, submitting `over_sampling_batch_size` at a time (SL/slime/rollout/sglang_rollout.py:503-559) | `rollout_fn` / `env_factory` per source; admission up to the cap | nothing |
| Dynamic filter | `dynamic_sampling_filter_path`; a dropped group decrements `remaining_batch_size` so more are submitted (`:530-546`) | `filter_zero_variance` evicts; `admit` refills (C/scheduler.py:172-178) | equivalent |
| Partial rollout | when B groups are kept, `abort()` cancels the rest. Groups with any aborted sample holding > 0 tokens are recycled (`:456-473`) to the data buffer (`:774` → SL/slime/data/data_source.py:220-233), which is served first. `start_rollout_id` is recorded but **never read**: no cap on how many rollouts a partial spans | colocated_partial: aborted calls are re-issued or resumed inside the same task (no rollout leaves the pool); `max_cuts` σ bounds retention; `wait`/`drop` bound τ | **Crucible beats.** [ran] `p13` below |
| `--mask-offpolicy-in-partial-rollout` | zero `loss_mask` on all previously generated tokens, so only the final segment trains (`sglang_rollout.py:295-297`) | none. Token mode trains every span under the correction | PATTERN, S: `TrainerConfig.train_spans="all"\|"last"` |
| Groups aborted with 0 tokens | not recycled: the prompt is dropped for this epoch (`:465-466`) | re-issued; never lost | nothing |
| TIS | `tis = exp(old − rollout)` clamped to [`tis_clip_low`=0, `tis_clip`=2]; `pg_loss *= tis` (SL/slime/backends/megatron_utils/loss.py:899-920); `icepop_function` mode `mis` (`:923-945`); `old` = actor-recomputed `log_probs` | `tis(C)` = `sg(min(r_dis, C))·ppo`: identical to AReaL clamp [0, 2] (`p1`, Δ = 0) | nothing. Crucible has no lower clamp (slime default 0, so no difference) |
| Aggregation | default: per-rollout token mean over `rollout_mask_sums`, then × `num_microbatches/step_global_batch_size` (`loss.py:1510-1563`, SL/.../cp_utils.py:54-99). `--calculate-per-token-loss`: Megatron divides by the global token count | `token_mean_global` default: each micro batch is scaled by its share of the global trained-token count (C/trainer.py:640), equal to slime per-token | Crucible's non-token modes (`seq_mean_token_mean`, `prompt_mean`) force a single micro batch with no token budget and no DP (C/trainer.py:606-607, 625), and average per *item* (a segment or call), not per rollout. P2 note for long agentic rollouts |
| Weight update | `update_weights`: rank 0 calls `pause_generation(mode="abort" if flush_cache else "in_place")`, flushes every `--flush-cache-interval` (SL/slime/utils/weight_sync.py:4-13), TP all-gather per param → HF convert → 512 MiB buckets (`--update-weight-buffer-size`, arguments.py:572-574) → **one** broadcast per tensor from rank 0 (DP0, TP0 per PP stage) into a group of all engine GPUs. Metadata goes over Ray, and `weight_version` is passed to SGLang (update_weight_from_distributed.py:103-142, 167-197, 325-357). Experts take a separate pass with EP all-gather, budget ÷ EP size (`:199-246`) | see §1.4. No TP/EP gather, no HF-name conversion, no expert pass, no `weight_version` sent to the server | PORT (M): stream per param → cast → bucket; DEPEND on SGLang's `/init_weights_update_group` + `/update_weights_from_distributed` as the receiver. Correction to the digest: slime sends to all engine ranks from one training rank per PP stage, not "rank 0 per node" |
| `in_place` pause | requests keep their KV across the update and continue on new weights. SGLang reports one `weight_version` per segment, so tokens before and after the update inside one segment share the final label | `resume_keeps_kv` capability, counted as `kv_resumes` (C/runtime.py:272-273); spans are always split at the resume | Crucible is stricter |
| Old-version files | `--update-weight-disk-keep-files` | `CheckpointPublisher` never deletes `v{n}/` (C/weights.py:64-67) and writes the fp32 master | PORT, S: keep the last K; cast to the serving dtype |
| R3 | `--use-rollout-routing-replay` consumed by Megatron (arguments.py:1207, 2183-2186) | `routing_replay="r3"` for `TinyMoEPolicy` only; HF/vLLM MoE capture open (M12-train.md §9 #36) | gap at real-MoE scale (L) |
| Fault tolerance | `--use-fault-tolerance`, `RolloutHealthMonitor` (SL/slime/utils/health_monitor.py:10), router worker pruning before abort (`sglang_rollout.py:436-445`) | `GatewayPool` health loop, evict and revive with replayed weights (C/gateway.py:100-121). With every engine down, calls return `abort` every 50 ms (`:126-129`). Nothing finishes, so the breaker never trips and the run ends on `wall_clock` | P2: an `engines_down` stop reason |
| Non-finite step | skipped (model.py:668-680) | not skipped | D7 |

[ran] `p13` simulates slime's partial rollout: over-sampling of 8 groups in flight, keep the first B=4, recycle the rest with their progress. It compares head staleness in rollouts against the Crucible sim.

| Workload | slime max head staleness | slime share > 2 | Crucible colocated_partial τ=2 wait | Crucible + `max_cuts=2` |
|---|---|---|---|---|
| tail 0.2×10 | 4 | 10.6% | max 2 | max 2 |
| tail 0.05×60 | **18** | 11.9% | max 2 | max 2, 4 retention evictions |

---

## 3. ROLL (`RO`, 581046a)

| Aspect | ROLL | Crucible | Verdict |
|---|---|---|---|
| Ahead-of-training | `async_generation_ratio`, an int in agentic despite its `float` annotation (RO/roll/configs/base_config.py:552-557, rollout_scheduler.py:277). It pre-creates `ratio` steps of groups | S·B·n cap | equivalent |
| Staleness | measured by group `create_step`. Expired groups are popped at `advance_step` (`rollout_scheduler.py:280-290`) and ignored at `get_batch` (`:537-541`), i.e. drop-at-consumption on the submit step. No per-token version anywhere | per-call versions and spans | Crucible beats (finer and exact) |
| Group redundancy | `group_size_redundancy`: run n+r members and take the first n to finish (`get_episode_id` `:294-324`, `put` `:330-349`, `get_batch` `:543-545`) | none | PATTERN (S), **with a length-bias guard**: the first n finishers are biased short. Crucible's `drop_early_short`/KS monitor (C/scheduler.py:179-187, C/sim.py:361) should gate it |
| Group filter | `group_filter_cls.filter(...)`; a filtered group is replaced with `advance_group(current_step)` (`:340-346`) | evict + refill | equivalent |
| Partial rollout | weight sync pauses, aborts and drains. Agentic discards the partial turn (research 02 §2) | re-issue or resume | Crucible beats |
| Loss | `ratio_type="segment"`: a geometric-mean ratio per contiguous response segment (per turn) (RO/roll/pipeline/agentic/agentic_actor_worker.py:47-51, utils.py:337-380); dual clip `1+2·pg_clip`; `train_infer_correction` IS weight and filter mask (`:31-45`, `:72-73`); `agg_loss` with global token and sample denominators (RO/roll/utils/functionals.py:412-452) | `gspo` is per item (C/mismatch.py:169-176). In segment items that spans every turn of a compaction segment | PATTERN, S: turn-level GSPO (`seq_ids` = call spans; already plumbed as `call_spans` in `bridges.common.flatten`) |
| Env-level masking | an all-`None` group exits; runner error → score 0 (research 02 §2.4) | M10 mask vs agent_broke_env | nothing |

---

## 4. Crucible's runtime against these three: invariants under adversarial schedules

[ran] `p6`: 17 schedules through `sim.simulate`, with `Scheduler.trained` wrapped to log every trained gid. n=4, B=4, 25 steps.

| Schedule | Steps | Max staleness (τ) | Over τ | Double-trained | Conservation |
|---|---|---|---|---|---|
| extreme tail 0.02×200, wait / drop / disaggregated | 25 / 25 / 25 | 2 / 2 / 2 (2) | 0 | 0 | ok |
| 30% all-masked groups + 10% masked members (pad / drop_group) | 25 / 25 | 2 (2) | 0 | 0 | ok (36 / 89 masked evictions) |
| 100% masked | 0 (runs to max_time) | n/a | n/a | n/a | expected; the runtime breaker stops it |
| token_budget Φ=50 (cut while every group is open) | 25 | 2 | 0 | 0 | ok |
| completion_frac λ=0.1 | 25 | 2 | 0 | 0 | ok |
| all long + max_cuts 0 / 1 | 25 / 25 | 1 | 0 | 0 | ok (96 / 9 retention evictions) |
| bound_overflow drop / allow (τ=1, S=3) | 25 / 25 | 1 / 2 | 0 / 351 (documented τ+1) | 0 | ok |
| **windowed_fifo + wait** | **0** | n/a | n/a | n/a | **deadlock (D3)** |
| greedy + filter_zero_variance | 25 | 2 | 0 | 0 | ok |
| per-source τ a=1, b=4 (disaggregated) | 25 | a 1, b 4 | 0 | 0 | ok |
| elastic + resume_keeps_kv | 25 | 2 | 0 | 0 | ok |
| publish_every=3, disaggregated | 25 | 2 | 0 | 0 | ok (note: τ counts publishes, so staleness in optimizer steps is up to K·τ) |

Runtime probes (`p7`-`p9`, real asyncio `Runtime` + TorchEngine bf16 + TorchTrainer):
- windowed_fifo deadlock (D3);
- refill livelock (D5);
- `max_offpolicy_frac` staleness 8 (D1);
- abort window (D10).

---

## 5. Weight sync: what Crucible lacks for 8×H100 and multi-node

| Need | slime / AReaL | Crucible today |
|---|---|---|
| Stream params without materialising the model | per-param all-gather → bucket → broadcast | full `state_dict` on rank-0 CPU (C/fsdp.py:163-171), then `.to(device)` per bucket (C/weightsync.py:276-278) |
| Ship the serving dtype | cast to bf16 before bucketing (AR fsdp_engine.py:1733-1736; slime `convert_to_hf`) | fp32 master shipped and written (C/weights.py:67) |
| TP / EP / PP sources | TP all-gather, EP expert pass with budget ÷ EP, one update group per PP stage (slime `:75-90, 199-246`) | FSDP/HSDP only; one owner, or owners by name |
| Feed SGLang/vLLM directly | native `update_weights_from_distributed` receivers with `weight_version` | `EngineReceiver` → in-process `load_weights(state)`; servers only via disk (`CrucibleWorkerExtension` loads the whole file on every TP worker, C/vllm_ext.py:16-23) |
| Persistent communicator | built once, destroyed on reconnect | `new_group` per subgroup update (D9) |
| Receiver memory | in place | staged 2× copy (C/weightsync.py:150-161) |
| Disk retention | `--update-weight-disk-keep-files` | unbounded `v{n}/` |
| Tied weights | HF conversion dedups | both keys cloned and shipped (C/trainer.py:202-203); correct but duplicate bytes |
| Version on the server | `weight_version` sent and echoed | client counter (D11) |

Crucible does better on three points:
- atomic all-or-nothing commit with all-reduced error codes (C/weightsync.py:256-266);
- a plan crc check (`:234-240`);
- a portable plan for elastic join.

AReaL and slime have none of these; a failed bucket there leaves engines half-updated.

---

## 6. Picks

| # | Pick | From | Verdict | Effort | Fixes |
|---|---|---|---|---|---|
| 1 | Detached-weight gating (`w = where(gate, r, 0).detach(); J = w·Â·logp`) + sanitise a non-finite log-ratio to 0 | AR functional.py:229-231, 420 | PORT | S | D7 |
| 2 | Skip the optimizer step on a non-finite grad norm; count `train/skipped_steps` and alarm | AR fsdp_engine.py:742-744; SL model.py:668-680 | PORT | S | D7 |
| 3 | Version stamped per emitted token (or append a span only on ≥ 1 token) | AR remote_inf_engine.py:1165-1167 | PORT | S | D6 |
| 4 | Fix the scheduler: apply `max_offpolicy_frac` after the at-bound rule and never to groups at their bound; count windowed-FIFO W over *untrained* groups; refuse W < B; cap `refills` (then `drop_group`) | own | n/a | S | D1-D3, D5 |
| 5 | Make the tests real: assert `r.steps == requested` in every sim test; a staleness ≤ τ assert for frac; a bound that can fail | own | n/a | S | D4 |
| 6 | `loglinear` π_prox approximation (no extra forward) | AR actor.py:1371-1452 | PORT | M | throughput |
| 7 | Pause (close admission) before abort at S1 | AR remote_inf_engine.py:1114, fsdp_engine.py:1695; SL update_weight_from_distributed.py:113-120 | PATTERN | S | D10 |
| 8 | Read SGLang `meta_info.weight_version`; pass `weight_version` on update | SL types.py:487-488, `:325-357` | PATTERN | S | D11 |
| 9 | Streaming per-param gather → bf16 → bucket → SGLang native distributed update, with a persistent group | AR fsdp_engine.py:1682-1772; SL update_weight_from_distributed.py | PORT (sender) + DEPEND (SGLang receiver API) | M | §5, D9 |
| 10 | `train_spans="last"` (mask older spans in token mode) | SL sglang_rollout.py:295-297 | PATTERN | S | option |
| 11 | Keep the last K checkpoint dirs; publish in the serving dtype | SL `--update-weight-disk-keep-files` | PORT | S | §5 |
| 12 | `group_size_redundancy` (first n of n+r) behind the length-bias monitor | RO rollout_scheduler.py:294-349 | PATTERN | S | long tail |
| 13 | Turn-level GSPO ratio (`ratio_type=segment`) | RO agentic_actor_worker.py:47-51 | PATTERN | S | option |
| 14 | OpenAI and Anthropic proxy with `InteractionCache` prefix tree for black-box agents | AR experimental/openai/{client,cache}.py | PATTERN | M | M12 §8 "AReaL proxy bridge" |
| 15 | KPop: decide whether the behave weight is applied; cite the source | AR functional.py:565-568 | PATTERN | S | D12 |
| 16 | `VLLMServerEngine`: send `resume_completion` or declare `resume: false` and refuse token mode | own | n/a | S | D8 |
| 17 | Document η ≡ S−1 and "`none` + S = AReaL mode" | AR staleness_manager.py:79-113 | doc | S | §1.1 |

Prior studies checked:
- `docs/research/m11-m12/02-areal-roll.md` §1.1-§1.6 matched the code on every point I re-read: the capacity formula, count-not-head bound, no consumption drop, creation-time FIFO, decoupled loss, bucket sizes, and that ROLL has no version.
- The zhihu digest is stale on AReaL's head-of-queue requeue and on staleness discard (§1.2).
- The zhihu digest's FSDP `update_weight_utils.py` path no longer exists in slime: `SL/slime/backends/` has only `megatron_utils` and `sglang_utils`.
