# M12 sweep, group 4: prime-rl, NeMo-RL, SkyRL, rLLM (and verifiers → prime-rl) vs Crucible M12 `train` (2026-10-12)

Read-only on `crucible/src` and `crucible/tests`. Crucible HEAD `3b522cb` (only `crucible/.coverage` modified).

**Roots**
- `C` = `crucible/src/crucible`.
- `REF` = `RL-Environments/06-frameworks-and-env-code`, which holds:

| Short | Repo | Pin | Licence |
|---|---|---|---|
| `PR` | `infra-primeintellect-ai-prime-rl` (configs in `PRC` = `PR/packages/prime-rl-configs/src/prime_rl/configs`) | f7e8ab6 | Apache-2.0 |
| `NR` | `infra-nvidia-nemo-rl` | c9193bc | Apache-2.0 |
| `SK` | `infra-novasky-ai-skyrl` | 3d0ae22 | Apache-2.0 |
| `RL` | `infra-rllm-org-rllm` | 4e828c8 | Apache-2.0 |
| `VF` | `primeintellect-ai-verifiers/verifiers/v1` | ae819de | MIT |

**Probes**
- Location: `/private/tmp/claude-501/-Users-muahmed-Desktop-Projects-RL-for-Real-World-Environments/54b8300a-e533-4d3f-9f31-c0f8369fa981/scratchpad/m12/g4/`. Each `probe_*.py` has a `.out` next to it.
- Upstream code is loaded **verbatim** by `upstream.py`. It extracts the named functions and classes with `ast` from the pinned checkouts and stubs only typing-only imports (jaxtyping, loguru, omegaconf).
- The NeMo-RL IS and KL blocks are executed as the literal source lines `loss_functions.py:713-825` and `:658-706`.
- `[ran]` means executed with `crucible/.venv/bin/python` (torch 2.14), on CPU, in float64 unless stated.
- No GPU, Docker or paid API was used.

**Method.** I read the Crucible loss, trainer, DP and bridge paths myself. Two read-only sub-sweeps read the async control of PR, NR and SK, and the VF → PR sample path. I re-opened their load-bearing citations, and all matched:
- `PR/orchestrator/orchestrator.py:87` (`TARGET_LAG = 1`)
- `PR/orchestrator/utils.py:42-61`
- `SK/skyrl/train/fully_async_trainer.py:1038,1061-1070` (warn only)
- `NR/nemo_rl/algorithms/grpo.py:3754-3761` (no `valid_mask`) together with `advantage_estimator.py:125`
- `VF/trace.py:620-632` (`is_truncated` is a plain property)
- `VF/trace.py:107-115,389` (`agent` is an `AgentInfo` dict)

**Severity**
- **P0**: wrong gradient or reward, or silent training on bad data.
- **P1**: a contract break, a crash, or a vacuous test.
- **P2**: a gap.

**Verdicts**
- **DEPEND**, **PORT** (copy, keep the Apache-2.0 header), **PATTERN**.
- Effort is S, M or L.

## Headline defects

| # | Sev | Crucible file:line | Defect | Evidence |
|---|---|---|---|---|
| H1 | **P0** | `C/train/mismatch.py:130-133` (dis), `:135-140` (mimo4), `:156-162` (hispo), `:110-122` (ppo/icepop with `old_logprob_source="rollout"`); `C/train/trainer.py:649-652` | **One extreme behaviour log-prob turns the whole model into NaN, silently.** The schema rejects inf and NaN (`allow_inf_nan=False`), but a *finite* behaviour log-prob ≤ −(709 − \|log π_θ\|) overflows `exp(logp − rollout_logp)` in float64. `torch.where(gate, r·…, 0)` then back-propagates `0·inf = NaN`. `TorchTrainer.step` clips with `clip_grad_norm_` (it returns NaN) and calls `opt.step()` with no finiteness check. | [ran] `probe_kl_edges.out` (b), on the per-token gradient with one token at behaviour log-prob −800 or −9999: NaN for dis, mimo4, hispo, icepop, ppo(src=rollout); finite for tis, cispo, kpop, mispo. In (c), the default `TrainerConfig()` (dis) and hispo through the real `TorchTrainer.step` give `train/loss=0`, `grad_norm=nan`, and **7/7 parameter tensors non-finite after one step**. Upstream guards it: SK `safe_exp_delta` clamps the log-ratio to ±20 (`torch_utils.py:188-193`); NR `nan_to_num(posinf=0)` (`loss_functions.py:720-733`); PR applies the keep-mask before `exp` (`loss.py:202-212`). −9999.0 is the floor that OpenAI-compatible servers report for −inf log-probs. That is not verifiable in this tree (no vLLM source), but nothing in Crucible clamps it. |
| H2 | **P1** | `C/train/trainer.py:712` (`_reduce_grads`) | **DP all-reduce of a rank-dependent bucket.** `grads = [p.grad for p in params if p.grad is not None]` is flattened into one buffer. A parameter that only one rank's items touch gives the ranks buffers of different sizes. Examples: a vision tower on a text-only partition (`train/vision.py` exists), an adapter, an untouched expert. Gloo aborts both processes; on NCCL a size mismatch is undefined (hang or misaligned sums). The empty-rank case is handled (`:645-648`), but this case is not. | [ran] `probe_none_grad.out`: a `TinyPolicy` with an adapter used only when the prompt contains token 47, 2 gloo ranks: `gloo::EnforceNotMet ... op.nread == op.preamble.nbytes`, both ranks SIGABRT. |
| H3 | **P1** | `C/train/trainer.py:534-535` | **The reference-KL term is unclamped, and it differentiates the forward KL.** `w·(exp(ref−lp) − (ref−lp) − 1)` with no input or output clamp: one token with log π_θ = −30 against log π_ref = −0.01 gives a value of 1.06e13 and a gradient of −1.06e13. NR clamps \|log r\| ≤ 20 and kl ≤ 10 (`utils.py:80-109`), giving 10 and 0 here. Under `grad_clip` that one token sets the direction of the whole step. Also, the exact expectation of Crucible's per-token gradient is ∇KL(π_ref‖π_θ) (forward), not the reverse KL that `kl_coef` suggests. NR's default k3 multiplies by `exp(lp − sg(lp))` (`loss_functions.py:677-679`), which recovers ∇KL(π_θ‖π_ref) exactly. | [ran] `probe_kl_edges.out` (a): crucible vs ∇forward 2.8e-17, vs ∇reverse 1.0e-1 (\|∇\|=0.30); NeMo vs ∇reverse 5.6e-17. (d): the magnitudes above. |
| H4 | **P1** | `C/train/bridges/adapters.py:29-41` | **The SkyRL bridge does not run in SkyRL's trainer.** No `rollout_metrics` key: `SK/skyrl/train/trainer.py:1069` raises `KeyError` before validation. A multi-segment (compacted) or multi-actor rollout gives more rows than prompts, which fails `validate_generator_output` (`trainer_utils.py:700-701`). Step-wise mode needs `is_last_step`/`trajectory_ids`, which are not emitted. `stop_reasons` are hard-coded to `"stop"`, so a truncated record shows `stop` and SkyRL's overlong filtering can never fire. This was already flagged as M11-sweep-G4 D10 and is still open at HEAD. | [ran] `probe_bridges.out`: KeyError in all 3 cases; `Mismatch between prompts (2) and responses (3)`; `is_last_step is missing`; `stop_reasons=['stop','stop']` for a `truncated=True` record. |
| H5 | **P1** | `C/train/bridges/adapters.py:50-56` | **The rLLM bridge trains nothing on rLLM's verl backend.** Crucible emits `Step(prompt_ids, response_ids, logprobs, advantage=list, reward)` with no `model_output`. rLLM's verl transform skips every step whose `model_output is None` (`RL/rllm/trainer/verl/transform.py:293-295`). Its `update_dataproto_with_advantages` coerces the advantage with `float(adv)` (`:595-597`), which raises on a per-token list. Only the tinker path (`RL/rllm/trainer/tinker/transform.py:100-112`) consumes the bridge's fields. `weight_version` is never set (Crucible has `policy_version`), so rLLM's async staleness is blind, and `Trajectory.reward` stays `None`. **This corrects M11-sweep-G4 §1.5 / "Not defects: rLLM Step mapping".** | [ran] `probe_bridges.out`: `model_output=[None, None]` → the skip gate is True for both steps; `float(list)` → TypeError; tinker `_collect_precomputed_advantages` → `[0.5, 0.5, 0.5]` (works). |
| H6 | **P1** | `C/interop/verifiers/v1_import.py:104,127-129` | **The verifiers v1 import feeding prime-rl-style traces loses truncation and actor identity.** (a) `trace.get("is_truncated")` is always falsy: `Trace.is_truncated` is a plain `@property` (`VF/trace.py:620-632`), so it is not in the dump. The stop list contains `max_tokens`, which verifiers never emits; it emits `max_input_tokens`/`max_output_tokens` (`VF/session.py:100-112`). The last call's `finish_reason == "length"` is not read. Length-cut traces are therefore recorded `truncated=False`, and `Turn.finish` is never set. (b) `actor=str(trace.get("agent"))` stringifies the whole `AgentInfo` dict (`VF/trace.py:105-115,389`), so `frozen_actors` and per-seat routing can never match it. (c) `agent.trainable`, `is_timeout` and episode-level `ok` are ignored. prime-rl drops timed-out and `episode.ok=False` traces from training (`PR/orchestrator/dispatcher.py:669-674`; `algo/base.py:11-18`). | [read]; the citations were re-opened. G5 D11 of the M11 sweep was only partly fixed. |
| H7 | P2 | `C/train/objectives.py:59-60`; `C/train/trainer.py:638` | **`const_max_len` is normalised by the batch's longest item, not a constant.** The trainer never passes `max_len`, so the HisPO/Dr.GRPO "constant" changes every batch, unlike SkyRL's `seq_mean_token_sum_norm` with `max_seq_len` (`ppo_utils.py:1201-1203`). | [ran] `probe_losses.out`: with L=T, equal to SkyRL (1.3e-9). The trainer default (batch max 12) against SkyRL with L=4096 differs by 340× in scale. |
| H8 | P2 | `C/train/trainer.py:606-607,625`; `:700-701` | **Non-token-mean aggregations run the whole global batch as one micro batch and cannot use DP.** `token_budget` is ignored, so memory is unbounded. `DistributedTorchTrainer` refuses them. NR passes `global_valid_seqs` (`clipped_pg.py:173-178`) and SK pre-scales advantages per mini-batch (`ppo_utils.py:1152-1220`; `trainer.py:1559-1577`), so every mode survives micro-batching and DP there. The critic, `teacher` (full-vocab OPD) and seq-mean are also single-process only: `DistributedTorchTrainer.__init__` takes no `critic` or `teacher`. | [ran] `probe_global_norm.out`: `seq_mean_token_mean`/`prompt_mean`/`const_max_len` with `token_budget=30` give `micro_batches=1`, 317 forward tokens. |
| H9 | P2 | `C/train/trainer.py:621-625`; `C/train/batching.py:6-7`; `docs/modules/M12-train.md:53` | **"mini batch = `updates_per_batch` slices of a global batch" is not what the code does.** Each update re-runs the *whole* global batch (PPO epochs). No disjoint mini-batch split exists (SK `policy_mini_batch_size`, NR `train_global_batch_size`). | [read] |
| H10 | P2 | `C/train/mismatch.py:116-122` vs `PR/trainer/rl/loss.py:193-224` | **Name clash: Crucible `icepop` ≠ prime-rl `icepop`.** prime-rl's IcePop is a masked REINFORCE-IS on π_θ/π_inference over [0.2, 5], and its gradient equals Crucible **`mimo4`**. Crucible `icepop` is NR's IS-weighted PPO. A prime-rl config ported by name trains a different objective. prime-rl's **default `ipo`** (absolute-probability trust region, `PRC/trainer.py:522-531,626`) has no Crucible counterpart. | [ran] `probe_losses.out`: mimo4 vs PR IcePop grad 1.4e-17; Crucible icepop vs PR IcePop grad 2.3e-2; dis or kpop vs PR IPO differ. |

## 1. Losses, numerically [ran] `probe_losses.py` → `probe_losses.out`

Setup:
- B=6, T=12 float64.
- Ragged mask with lengths 12, 7, 3, 9, 1, 5.
- π_θold = π_θ + 0.25·N; π_rollout = π_θold + 0.4·N.
- Per-token Â (sequence value + 0.1·N).
- Loss = −`aggregate(correct(...))`; the gradient is w.r.t. the per-token log π_θ leaf.

SkyRL computes ratios in float32 (`safe_exp_delta`), hence its ~1e-9 floor. Crucible's loss path is float64 (`trainer.py:673-675`).

| Crucible (`mismatch.py` + `objectives.py`) | Upstream (verbatim) | \|Δloss\| | max\|Δgrad\| | max\|grad\| | Verdict |
|---|---|---|---|---|---|
| ppo(0.2, 0.28), token_mean_global | NR `clipped_pg_actor_objective` token-level (`clipped_pg.py:116-179`) | 9.9e-11 | 8.9e-12 | 3.3e-2 | equal |
| ppo + dual_clip 3, with r up to 7.4 (13 tokens hit) | NR `ratio_clip_c=3` | 2.2e-10 | 8.5e-12 | 3.2e-2 | equal (sanity: without dual clip, Δgrad 0.24) |
| ppo + dual_clip 3, with r up to 7.4 | SK `dual_clip` (`ppo_utils.py:557-587`) | 3.6e-9 | 3.0e-9 | 3.2e-2 | equal |
| ppo(0.2, 0.28) | SK `regular`, `token_mean` | 2.6e-9 | 3.0e-9 | 3.3e-2 | equal |
| tis C=2 | NR use_IS + `tis` (`loss_functions.py:746-765`) | 1.3e-10 | 1.7e-11 | 6.3e-2 | equal |
| tis C=2 | SK regular + token TIS 2.0 (`off_policy_correction_utils.py:54-60`) | 6.9e-9 | 5.9e-9 | 6.3e-2 | equal |
| icepop(0.5, 5) | NR use_IS + `icepop` (`:766-785`) | 1.4e-10 | 2.1e-11 | 7.7e-2 | equal |
| cispo(0.2, 0.28) | NR `use_cispo` | 1.7e-10 | 8.9e-12 | 3.3e-2 | equal |
| cispo(0.2, 0.28) | SK cispo, anchor old (`:733-821`) | 5.2e-10 | 1.1e-9 | 3.3e-2 | equal |
| cispo m2, eps_high=4, src=rollout | PR `CISPOLoss(0, 5)` (`loss.py:256-285`) | 0 | 0 | 7.7e-2 | equal |
| ppo(0.2, 0.2), src=rollout | PR `PPOLoss(0.8, 1.2, cap 1e4)` (`:227-253`) | 0 | 0 | 7.7e-2 | equal (PR caps the ratio *value* at 1e4 with a straight-through gradient; Crucible has no cap) |
| dis(0.8, 3.0) | SK `rollout_is(0.8, 3.0)` (`:825-863`) | 2.5e-9 | 3.7e-9 | 6.9e-2 | equal |
| mimo4 [0.2, 5] both signs | PR `IcePopLoss(0.2, 5)` (`:193-224`) | value differs by design | 1.4e-17 | 7.7e-2 | gradient equal (see H10) |
| gspo(3e-4, 4e-4), seq_ratio | SK `gspo`, `sequence_mean` (`:664-729`) | 0 | 1.7e-18 | 1.3e-2 | equal (GSPO-token) |
| gspo, seq_ratio, sequence-constant Â | NR GSPO (seq IS, `token_level_loss=False`) | 1.7e-9 | 6.6e-11 | 1.2e-2 | equal |
| gspo, seq_ratio, per-token Â | NR GSPO | 1.6e-9 | 4.5e-3 | 1.1e-2 | differs by design: NR differentiates the *sequence* ratio, so each token gets the mean Â/\|o\|; Crucible and SK use GSPO-token |
| gspo, token_mean_global (the default agg) | NR GSPO | 7.9e-2 | 1.4e-2 | 1.1e-2 | P2: Crucible does not warn on the wrong agg for gspo (SK warns, `:683-692`) |
| gspo, seq_ratio | NR GSPO + `use_importance_sampling_correction` | 3.7e-1 | 1.5e-2 | 2.2e-2 | P2 gap: Crucible gspo has no π_θold/π_rollout term (NR multiplies by `exp(Σ(prev−gen))`, `:717-725`) |
| ppo, seq_mean_token_mean | NR `token_level_loss=False` | 1.3e-9 | 8.5e-10 | 7.3e-2 | equal |
| ppo, seq_mean_token_mean | SK `sequence_mean` | 7.2e-10 | 6.0e-9 | 7.3e-2 | equal (edge: SK divides by *all* rows incl. zero-mask rows, `:1199`; Crucible drops them) |
| ppo, prompt_mean | SK `prompt_mean` | 1.3e-9 | 1.7e-9 | 2.6e-2 | equal |
| ppo, const_max_len (L given) | SK `seq_mean_token_sum_norm` | 1.3e-9 | 1.3e-9 | 1.7e-2 | equal; the trainer never gives L (H7) |
| dis or kpop | PR `IPOLoss(eps 0.3)` (default) | 1.1 / 0.11 | 1.2e-2 / 7.7e-2 | — | no counterpart (H10) |

**Advantages [ran]** (groups of 4, 3 and 1):

| Estimator | Crucible | Upstream | Agreement |
|---|---|---|---|
| GRPO+std | `[1, −1, −1, 1, 1.414, −.707, −.707, 0]` | rLLM `rl_algo.py:6-18` | identical (population std) |
| GRPO+std | same | SK `ppo_utils.py:1344-1393` | sample std (Bessel): `[.866, …, 1.155, −.577, …]`. A configuration difference, not a bug. |
| RLOO | (all three) | rLLM, SK | equal |

- **Singleton group:** Crucible gives 0. rLLM GRPO gives the raw reward (mean 0, std 1), and rLLM RLOO returns the rewards too.
- **NR baseline:** leave-one-out by default. Masked rows (`loss_multiplier=0`) still vote in the baseline and std (`grpo.py:3754-3761` and `5556-5572` pass no `valid_mask`; `advantage_estimator.py:125`). Crucible excludes masked rows (`objectives.py:91-93`), which is correct. Any future NR export must not rely on NR to do it (this agrees with the M11 G3 D7 note at `C/interop/nemo_gym/server.py:26-27`).

**Upstream objectives missing in Crucible**
- PR `ipo` (the default)
- SK `sapo` (soft sigmoid gate)
- SK `dppo` binary-TV/KL, sign-aware. Crucible `kpop` is symmetric and ignores the sign of Â.
- SK `clip_cov`/`kl_cov`
- SK sequence-level TIS and product/geometric sequence masks and outlier-token masks
- NR `seq-mask-tis` (geometric gate with raw token IS weights). Crucible `mispo` and `hispo` are different functions.
- NR `force_on_policy_ratio`
- NR positive-example NLL (VAPO)

All are pure functions of the same tensors (see picks).

## 2. Global normalisation across micro batches and DP ranks [ran] `probe_global_norm.py`

**Setup**
- 12 multi-turn records (1-3 calls each, so segment items carry observation tokens).
- The gradient is captured just before `AdamW.step` (`grad_clip=1e9`). Parameters are not compared.
- Three runs per case: 1 micro batch (budget 1e9), many micro batches (budget 30), and `DistributedTorchTrainer` on 2 gloo ranks (budget 30).

| Case | 1 mb vs many | 1 mb vs 2 ranks | max\|g\| | micro batches (1/many/dp) |
|---|---|---|---|---|
| dis | 4.5e-10 | 4.4e-10 | 2.4e-3 | 1/11/11 |
| ppo(recompute_current) | 6.5e-9 | 4.7e-9 | 2.5e-2 | 1/11/11 |
| cispo(rollout) | 3.7e-9 | 3.7e-9 | 2.0e-2 | 1/11/11 |
| gspo(rollout) | 2.3e-10 | 1.8e-10 | 2.2e-3 | 1/11/11 |
| dis + kl_coef 0.1 (k3 on every token) | 6.0e-8 | 6.7e-8 | 2.9e-1 | 1/11/11 |
| dis + OPD token (0.5) + rl_coef 0.5 | 6.1e-10 | 5.8e-10 | 3.9e-3 | 1/11/11 |
| dis, item_unit=call | 5.5e-10 | 5.8e-10 | 2.4e-3 | 1/21/22 |
| dis, updates_per_batch=2 (2nd update) | 1.1e-9 | 7.0e-10 | 2.2e-3 | 1/11/11 |

**Verdict: token_mean_global is a global token mean across micro batches and ranks, for the policy term, KL and OPD (float32 parameter floor).**
- Masked tokens (gates) stay in the denominator. PR's `rl_scale` from `loss_mask` (`train.py:311-327`) and NR's `global_valid_toks` do the same.
- SkyRL's KL and entropy terms are *not* global: they are a per-micro-batch mean weighted by `len(mb)/len(data)` and averaged by FSDP (`SK/workers/worker.py:1022,1245`). Crucible's KL is better here.
- Not covered: seq-mean, prompt-mean, const, critic and full-vocab OPD under DP, because they do not exist there (H8). The critic loss is a global token mean in one process (`trainer.py:254-262`); SK's is 0.5·sequence-mean with `value_clip` (`ppo_utils.py:156-174`).
- The existing DP and FSDP tests compare parameters after one AdamW step. I checked whether that hides a scale error. It does not: a 2×-wrong gradient moves parameters by 1.4e-3, far above the 6e-7 test tolerance.
- FSDPTrainer was not re-run here. It inherits the same loss path and differs only in `_reduce_grads`/`_equalize` (`fsdp.py:123-147`), so H2 does not apply to it: FSDP2 reduce-scatters every sharded parameter.

## 3. Async off-policy control

### 3.1 What the three frameworks do (sub-sweep, citations re-opened where marked ✔)

| | prime-rl | NeMo-RL async GRPO | SkyRL fully-async | Crucible |
|---|---|---|---|---|
| Bound | `max_async_level` **no longer exists**. `TARGET_LAG=1` dispatch and ship gate ✔ `orchestrator.py:84-87,599-611,963-987`; `max_off_policy_steps=8` (`PRC/orchestrator.py:591`) | `max_trajectory_age_steps=1` (`grpo.py:273-287`); `[v−age, target]` validity (`replay_buffer.py:945-951`) | `max_staleness_steps=4` (`config.py:1028-1051`) | `max_staleness` τ=4, `admission_staleness` S (`scheduler.py:44-73`) |
| Staleness unit | ✔ `(step−1) − policy.start`, oldest version, queue time included (`utils.py:48-61`) | generation version **at batch start**, even when in-flight updates change the weights (`trajectory_collector.py:842`) | `global_step − global_step_when_scheduled` (✔ `:1038`) | `v_train − oldest version touched` (per token via `version_spans`) |
| Enforcement | hard: `TrainSink._drop_stale` voids queued traces, and a stale cancel voids the whole group (`train_sink.py:161-198,230-241`); the in-flight cancel saves compute (`dispatcher.py:421-439`) | evict at sample time; training stalls until gap-filled (`replay_buffer.py:571-629`; `grpo.py:5224-5292`) | **soft**: AReaL capacity `(S+step)·mini − (accepted+running)` (`:181-187`); over-stale groups are **only warned** ✔ (`:1061-1070`) | `wait` (block), `drop` (evict terminal), `none`; admission S·B·n counts in-flight + buffered |
| In-flight at publish | `pause?mode=keep`, so requests continue on new weights; `PolicySpan{start,end}` per episode (`clients.py:403-410`; `dispatcher.py:720`) | default: drain on old weights; `in_flight_weight_updates`: keep-pause; `recompute_kv_cache…` optional (`trajectory_collector.py:1048-1158`) | keep-pause, no abort, no re-prefill (`remote_inference_client.py:207-225,1249-1251`) | partial: abort + re-issue (`call`) or resume with spans (`token`); disaggregated: finish on the pinned version |
| Loss requirement | ratio vs inference log-probs always (no π_θold) | `assert use_importance_sampling_correction` (`grpo.py:4724-4726`) | rollout-anchored loss required (`:393-404`) | default `dis` (rollout-anchored); π_θold corrections refuse multi-version rollouts (`trainer.py:611-615`) |
| Errors | dropped, never 0 (`algo/base.py:11-18`); train timeouts become errors (`dispatcher.py:669-674`) | env exception: the group is lost (first failure fatal by default, `max_generation_failures=0`); generation exception: `print; break`, partial reward kept, **not masked** (`rollouts.py:1370-1372`) | — | masked (`reward=None`), excluded from baselines |
| Checkpoint | orchestrator saves progress + sampler state only; buffer and in-flight work lost (`ckpt.py:30-47`) | — | — | barrier checkpoints; in-flight rollouts restart (`state.py`) |

### 3.2 Reproduced in `crucible.train.sim` [ran] `probe_async_sim.out`

Real `Scheduler`/`DataPool`, n=4, B=4, 30 steps, 4000 prompts, long-tail shapes.

| Strategy | Rule it mirrors | max staleness (colocated / disagg / elastic) | rollouts > τ | Evicted stale |
|---|---|---|---|---|
| `wait` τ=1 | Crucible default | 0 / 1 / 1 | 0 | 0 |
| `wait` τ=2 | — | 2 / 2 / 2 | 0 | 0 |
| `drop` τ=1 | prime-rl `max_off_policy_steps` | 0 / 1 / 1 | 0 | 0 |
| `drop` τ=2 | prime-rl | 2 / 2 / 2 | 0 | 12 / 27 / 27 |
| `none`, S=2 | SkyRL capacity bound only | **5 / 6 / 6** | 52 / 132 / 132 | 0 |
| `none`, S=4 | SkyRL | **13 / 12 / 11** | 114 / 168 / 164 | 0 |

- `inflight_violations` is 0 everywhere, and max in-flight equals S·B·n exactly.
- **Invariants hold.** `wait` and `drop` never train over τ. A capacity-only bound (SkyRL's design) does **not** bound staleness under long tails, which reproduces SkyRL's own "Staleness control violated" warning path. The prime-rl drop rule and Crucible `drop` agree on the definition. That holds because prime-rl trains batch `step` on v{step−1}, so `(step−1) − start` = `v_train − oldest`.

**Gaps (P2)**
- Crucible `drop` evicts only *terminal* groups (`scheduler.py:159-166`). A doomed in-flight group runs to completion first. prime-rl cancels it before the weight swap (`dispatcher.py:401-439`). PATTERN, S.
- NeMo-RL's "age at batch start" under-counts the staleness of rollouts that span updates. Crucible's per-token `version_spans` is stricter; keep it.
- prime-rl, NeMo-RL and SkyRL all keep-pause (requests continue on new weights, possibly on a stale KV). Crucible's colocated partial aborts and re-issues or resumes. Both are valid; Crucible records the spans.

### 3.3 Step-wise / TITO (SkyRL, NeMo-RL)

- **SkyRL step-wise** computes advantages on the last step and broadcasts them by `cumsum(is_last_step)` (`trainer.py:1212-1250`). `merge_stepwise_output` merges a turn only when `_is_prefix` holds, else it starts a new row silently (`generators/utils.py:855-978`).
  - Crucible's `bridges.common.flatten` (`common.py:58-72`) is the same merge, with an exact prefix check and a split on `prefix_break`. Equivalent.
  - In both, `sequence_mean` counts each step or segment as a "sequence". So Crucible's `seq_mean_token_mean` averages over **segments/calls, not trajectories** (`trainer.py:460-481`), and `gspo`/`mispo`/`hispo` "trajectory" gates are per row (`mismatch.py:141-171`). P2, document it.
- **SkyRL TITO** keeps token ids by construction. On the step-wise path it re-templates text every turn, so prefix consistency is only checked afterwards (`skyrl_gym_generator.py:465-474`). NeMo-RL tokenises observations without a chat template (`rollouts.py:1072-1074`, TODO). Crucible's exact TITO (`chat.py`), with `prefix_break` as a training mask, is stricter than both.

## 4. Bridges against current upstream APIs [ran] `probe_bridges.out`

| Bridge | Upstream contract | Runs? | Gaps |
|---|---|---|---|
| SkyRL `skyrl_generator_output` (`adapters.py:29-41`) | `GeneratorOutput` TypedDict (`SK/skyrl/train/generators/base.py:40-69`); `trainer.py:1069` indexes `rollout_metrics`; `trainer_utils.py:680-775` | **No** (H4) | Also: per-token rewards put the outcome on the last sampled token and drop Crucible's turn credit. That is deliberate for GRPO, but it means `token_rewards=True` + `grpo` sums to the outcome only. Zero-filled observation log-probs are fine (`loss_mask` 0), but missing completion log-probs are also zero-filled (`common.py:78-79`), which fakes p=1 behaviour (see §5). |
| rLLM `rllm_steps` (`adapters.py:50-56`) | `rllm.types.Step` (`RL/rllm/types.py:100-136`) | `Step(**d)` validates. verl backend: **no** (H5). tinker: yes | Not set: `weight_version`, `done`, `Trajectory.reward`, and the grouping (`TrajectoryGroup`/`Episode.task_id`). In broadcast mode without precomputed advantages, rLLM asserts that `traj.reward` is not None (`advantage.py:209`). `reward` is set on the last call of *each segment*, so multi-segment rollouts count the outcome twice if summed. |

## 5. verifiers v1 → prime-rl vs Crucible's import (sub-sweep, verified at H6)

**What prime-rl trains on**
- **Samples:** one sample per trainable root-to-leaf branch of `Trace.nodes`, with shared sampled nodes trained once (`PR/orchestrator/trajectories.py:105-191`).
- **Masks and log-probs:** `loss_mask` = the node mask, and `inference_logprobs` = the vLLM sampling log-probs.
- **Side streams:** per-token `rl_weights`/`ce_weights`/`ref_kl_weights` (`transports/batch/types.py:62-70`), `sampling_mask` (required when top-p/top-k truncate: `train_sink.py:263-271`), `routed_experts [tokens, layers, topk]`, and `temperatures`.
- **Advantages:** `reward − mean` over survivors, with no std (`algo/grpo.py:26-61`). Zero-advantage tokens get `rl_weight=0`.
- **Truncation:** a metric only; truncated traces still train.

| Gap in Crucible (import or schema) | Sev | Where |
|---|---|---|
| Truncation not detected (`is_truncated` absent from the dump; `max_input_tokens`/`max_output_tokens`/`finish_reason=="length"` missed); `Turn.finish` never set | P1 | `v1_import.py:127-129,104` |
| `actor` = repr of the `AgentInfo` dict; `agent.trainable` ignored; multi-trace episodes (solver + untrainable judge or user-sim) discarded whole instead of training the solver | P1 | `v1_import.py:104`; `vf_rollout` `:234-236` |
| `is_timeout` and episode `ok`/`errors` ignored (prime-rl drops both for training) | P2 (Crucible's own semantics score limit stops; decide and document) | `v1_import.py:110-126` |
| top-p/top-k: the generate server forwards only `max_tokens`/`temperature` (`v1_import.py:159`) and returns no support; `Turn.sampling` is never written, so `candidates_valid()` passes vacuously | P1 (silent sampling-mode change) | `v1_import.py:155-168` |
| `Turn.routing: list[list[int]]` holds one MoE layer; prime-rl and SkyRL carry `[tokens, layers, topk]` | P2 | `C/core/schema/rollout.py:45` |
| No per-token loss-weight streams (observation CE "echo", ref-KL-only tokens); no `is_content` | P2 | schema |
| Compaction or multiple branches → the whole record is masked (`v1_import.py:76-78`), where prime-rl trains each trainable branch | P2 | — |
| **Trainer accepts token-incomplete records.** `TorchTrainer.step` never calls `valid_for_training()`, and `flatten` pads missing log-probs with 0.0 (`common.py:78-79`). Only the M10 record path checks (`C/runtime/record.py:102-106`). A bridge or import that hands records straight to the trainer trains with π_rollout = 1 on those tokens. | P1 | `trainer.py:600`; `common.py:78-79` |

## 6. Picks

| # | Take | From | Mode | Effort |
|---|---|---|---|---|
| 1 | Clamp the log-ratio before every `exp` in `mismatch.correct` (±20 like SK, or apply the gate before `exp` like PR), plus a non-finite guard in `TorchTrainer.step` (skip the update, count it, alarm) | SK `torch_utils.py:188-193`; PR `loss.py:202-212`; NR `loss_functions.py:730-733` | PATTERN | S |
| 2 | `_reduce_grads`: `zeros_like` for every `None` grad before flattening, in a fixed parameter order | own (PR/SK rely on FSDP/DDP buckets) | own | S |
| 3 | Reference KL: input clamp ±20, output clamp 10; the k3 score-function weight `exp(lp − sg lp)` (reverse-KL gradient), with an `on_policy_kl_approximation` option weighting by π_θ/π_rollout | NR `utils.py:50-109`; `loss_functions.py:658-706` | PORT (Apache-2.0) | S |
| 4 | Fix the SkyRL bridge: all keys incl. `rollout_metrics`, `trajectory_ids`, `is_last_step` (step-wise for split rollouts), real `stop_reasons` from `truncated`/`Turn.finish` | SK `generators/base.py:40-69`; `trainer_utils.py:680-851` | TARGET | S |
| 5 | Fix the rLLM bridge: emit `ModelOutput`-backed steps (`Step.from_model_output`), a scalar advantage for the verl backend (per-token only for tinker), and set `weight_version`, `Trajectory.reward` and the grouping | RL `types.py:225-239`; `verl/transform.py:293,595`; `tinker/transform.py:100-112` | TARGET | S |
| 6 | verifiers import: truncation set plus last-call `finish_reason`; `Turn.finish`/`Turn.sampling`; `agent["name"]`; honour `trainable`; `is_timeout`/episode `ok` policy; forward top_p/top_k or refuse | PR `monitors/file/traces/index.py:35-45`; `algo/base.py:11-18` | PATTERN | S |
| 7 | Global sequence and prompt normalisers so every `loss_agg` runs in micro batches and DP: pre-scale per item (SK) or pass `global_valid_seqs` (NR); `const_max_len` takes the configured L | SK `ppo_utils.py:1152-1220`; NR `clipped_pg.py:169-178` | PATTERN | M |
| 8 | Add the missing objectives as pure corrections: PR `ipo` (default), SK `dppo` (sign-aware binary TV/KL), `sapo`, NR `seq-mask-tis`, SK sequence TIS / geometric / product / outlier masks; rename or alias `icepop` to make the PR mapping explicit (PR icepop = Crucible mimo4) | PR `loss.py:150-190`; SK `ppo_utils.py:591-660,867-935`; `off_policy_correction_utils.py:24-254`; NR `loss_functions.py:786-814` | PORT | M |
| 9 | `drop` strategy also cancels doomed in-flight groups before a publish, and voids the whole group | PR `dispatcher.py:401-439`; `train_sink.py:230-241` | PATTERN | S |
| 10 | Trainer-side `valid_for_training()` gate (refuse or mask token-incomplete records); `flatten` refuses missing log-probs instead of padding 0.0 | own (`C/runtime/record.py:102-106`) | own | S |
| 11 | Disjoint mini-batches (`updates_per_batch` slices) or fix the doc | SK `policy_mini_batch_size`; NR `train_global_batch_size` | PATTERN | S |
| 12 | Schema: `routing` per layer; optional per-token loss-weight streams (ce/ref_kl) for echo/OPD-only tokens | PR `transports/batch/types.py:46-70` | PATTERN | M |

**Not defects (checked).** Every Crucible correction that names an upstream counterpart is numerically equal to it, loss and gradient, at ≤ 1e-8. That covers: ppo, dual clip (engaged on 13 tokens), tis, icepop (NR), cispo (NR/SK/PR), dis (= SK rollout_is), mimo4 (= PR IcePop gradient), gspo (= SK GSPO-token; = NR GSPO for sequence-constant Â), seq-mean, prompt-mean, and const with a given L. Also checked:
- `token_mean_global` is exactly global over micro batches and gloo DP for the policy, KL and OPD terms.
- `wait`/`drop` hold τ in every topology.
- RLOO and population-std GRPO match rLLM.
- Masked rollouts are excluded from baselines, unlike NR `grpo.py`.
