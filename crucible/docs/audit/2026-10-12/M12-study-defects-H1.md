# M12 study H1: defects in the learning math of `train`

Scope: `C/train/{mismatch, objectives, trainer, critic, credit, batching, hf, sampling, group_ops, teachers, weights (QDQ), vision}.py`, `C/credit/{gar, shaping, hack_detector, signed}.py`, and what they call: `C/train/bridges/common.flatten`, `C/core/schema/rollout.py`, `C/train/vllm_http._nuclei`, `C/train/runtime.py` (group stage and teacher gate). Tests read: `tests/unit/test_train_{math,critic,opd,segments,vision,audit}.py`, `test_infra_algorithms.py`, `test_m6_audit.py`, `test_vllm_engine.py`.

Spec: `docs/modules/M12-train.md` (§1 planes, §2 units, §5, §7, §9 checklist) and `docs/research/m11-m12/06-article-systems.md` §5 (formulas for every correction).

Paths: `C` = `crucible/src/crucible/`. Probes: `$S` = the session scratchpad `m12/h1/` (`r01_*.py … r18_*.py`, helper `common.py`). Run from `crucible/` with `.venv/bin/python $S/rNN_*.py`.

Method: read-only study on HEAD `3b522cb`, CPU only (torch 2.14, transformers 5.19), offline. **[ran]** = reproduced by a probe; **[read]** = from the code alone. No repo file other than this document was changed.

## 0. What holds (checked, no defect)

- **Corrections match the formulas** (art. §5.3): DIS, mimo4, CISPO and the TIS / IcePop weights are detached; the IcePop and KPop gates use r_dis (π_θold vs π_rollout), the PPO term r_stale; HisPO clips the full ratio with three bounds; MIS-PO has no IS weight; PPO dual clip only bites for Â < 0. GSPO-token aggregated with `seq_ratio` equals the sequence objective min(s·Â, clip(s)·Â) exactly, value and gradient, clipped and unclipped (`r17`).
- **Micro-batching** with `token_mean_global` equals the one-batch gradient for all ten corrections, with a masked rollout, multi-call segment items and the k3 KL term on (max |Δgrad| ≤ 7.5e-9 at |g| ≈ 0.1; `r06`).
- **Group advantages**: masked members excluded; singleton and zero-variance groups give 0; RLOO, Dr.GRPO; `credit/gar.gar` conserves the passing mass for any λ.
- **Aggregations** `prompt_mean` and `const_max_len` equal hand computations (`r17`).
- **HFPolicy**: the LM head runs in fp32; positions `len(prompt)-1 …` predict the completion; right padding is correct.
- **MXFP4 QDQ**: block size 32 with power-of-two scale 2^(⌊log2 amax⌋−2) follows the OCP rule, and re-quantizing is idempotent in fp32 and bf16 (`r11`).

## 1. Findings

| id | sev | file:line | defect | evidence |
|---|---|---|---|---|
| V1 | P0 | `C/train/group_ops.py:92` | `gar` makes rewards binary at `pass_threshold=1.0`. The operators the module says to put before it (GRS, `bt_pairwise`, `tournament`) produce rewards in [0, 1], so GAR treats every passing solution scored below 1 as a failure. | `r03` [ran] |
| V2 | P0 | `C/train/mismatch.py:122,133,140` (+ `trainer.py:650`) | A token that the gate drops still back-propagates `0·r`. When r overflows, that is 0·inf = NaN. A behaviour log-prob of −inf or NaN (or < −88 in the fp32 loss dtype used on MPS) turns every parameter into NaN. No finiteness check runs anywhere: `valid_for_training` passes, there is no NaN guard on grad norm, and the step is applied. | `r05` [ran] |
| V3 | P1 | `C/train/trainer.py:52,513-515` | `TrainerConfig.temperature` is never compared with `Turn.sampling["temperature"]`, and the runtime does not pass its own temperature to the trainer. A run sampled at 0.6 with the trainer left at 1.0 trains on a biased ratio without any error. | `r02` [ran] |
| V4 | P1 | `C/train/trainer.py:460-481,606,638`; `objectives.py:44-61` | `seq_mean_token_mean`, `seq_ratio` and `const_max_len` count trainer **items**, not rollouts. A call with `top_p<1` or R3 switches the rollout to per-call items, and a compaction segment or extra actor adds items. The same data then gets a different loss. | `r01` [ran] |
| V5 | P1 | `C/train/weights.py:150-158` | `QDQ(model=HFPolicy)` never writes the grid into the trainer. Parameters are looked up by `HFPolicy.named_parameters()` names (`model.` prefix), but the published state dict uses HF names. The trainer and engine weights diverge with no error. | `r04` [ran] |
| V6 | P1 | `C/train/trainer.py:511-518`; `vision.py:111-117` | `TorchTrainer` never passes `Turn.media`, so `VisionHFPolicy` falls back to a text-only forward over the image placeholders. Vision records train against the wrong π_θ, and nothing refuses them. | `r08` [ran] |
| V7 | P1 | `C/train/trainer.py:431-432,525-535`; `teachers.py:45-56` | Token OPD and the k3 KL compare teacher or reference log-probs taken at temperature 1 over the full vocabulary with behaviour (and π_θ) log-probs taken at the sampling temperature and renormalized over the nucleus. With the teacher (or reference) **identical** to the policy, the signal is not 0. | `r10` [ran] |
| V8 | P1 | `C/train/runtime.py:201` | The teacher vocabulary gate runs only if `RuntimeConfig.student_vocab_id` is set, and the default is `None`. A teacher with a different vocabulary distils by default. The doc (§1) says vocabularies are "checked before any rollout". | `r14` [ran] |
| V9 | P1 | `C/train/trainer.py:319,328` | `speed_gate` counts any reward > 0 as correct. A failure that carries partial credit (κΦ = 0.18) earns the speed bonus and helps set the group's speed baseline. M6 moved GAR and verbosity to `pass_threshold=1.0`, but this gate was not moved. | `r13` [ran] |
| V10 | P1 | `C/train/vllm_http.py:145-151` | `_nuclei` rebuilds the nucleus by summing the top-k **processed** probabilities up to `top_p`. vLLM's `processed_logprobs` are taken after top-k/top-p, renormalized over the nucleus. The sum therefore reaches `top_p` before the last member, so the rebuilt set is too small: either the call fails, or the trainer silently renormalizes over the wrong set. | `r16` [ran, pure function; vLLM semantics from its `--logprobs-mode` doc, not run] |
| V11 | P2 | `C/train/trainer.py:217-238,406-407,441` | Some advantage sources are silently dropped. `credit_unit="reward_to_go"` replaces group-stage (GAR) advantages and ignores `estimator` / `std_normalize`. Critic GAE uses the raw `r.reward` (speed and latency shaping and GAR are ignored). The signed format penalty does nothing in critic mode. | `r09` [ran] |
| V12 | P2 | `C/train/trainer.py:236` | The truncation bootstrap uses `v[-1]` = V(state **before** the last token) as V(s_T). With γ = 1 the last TD error is always r, whatever the critic says. The suite's test uses a constant V, so it cannot see this. | `r07` [ran] |
| V13 | P2 | `C/train/trainer.py:606-607,625` | Every `loss_agg` except `token_mean_global` runs the whole global batch as one forward and backward, ignoring `token_budget`. This is a memory cliff at real scale, although the other aggregations decompose exactly. | `r15` [ran] |
| V14 | P2 | `C/train/hf.py:91-96` | `HFPolicy` computes `hidden @ lm_head` itself and skips what the model's forward does after the head (Gemma-2/3 `final_logit_softcapping`, Granite `logits_scaling`, Cohere `logit_scale`). Its log-probs then differ from the model's and the engine's. | `r12` [ran] |
| V15 | P2 | `C/train/weights.py:121` | MXFP4 element rounding sends ties to the lower magnitude (0.75→0.5, 1.75→1.5, 3.5→3), not round-half-to-even. The grid then differs from a spec quantizer applied to the master weights. | `r11` [ran] |
| V16 | P2 | `C/train/group_ops.py:52-55,124-130,140-154` | `zero_hacks`, `bt_pairwise` and `tournament` write a reward onto members with no reward and onto `mask_sample` members. Group-stage members include M10-masked World records (status `finished`, reward None). A tournament's win shares also count matches against masked members. | `r18` [ran] |
| V17 | P1 | tests | Some tests are vacuous and some branches are untested. Details are in §3. | §3 |

## 2. Details

### V1 (P0): GRS, BT or tournament followed by GAR turns passing solutions into failures

`group_ops.gar` starts with `rs = [1.0 if reward >= pass_threshold else 0.0 …]`. This rule is right for raw composed rewards (the M6 audit test `test_S_contracts_with_trainers_and_group_ops` relies on it). But the module docstring prescribes "hack zeroing → rubric (GRS) or comparison (BT, tournament) → GAR", and each of those writes a reward in [0, 1]. `r03`, with two tests-passing solutions and two failures:

- grs→gar, rubric (1.0, 1.0) and (0.9, 0.95): rewards [1.0, 0.855, 0, 0] give advantages **[0.75, −0.25, −0.25, −0.25]**. The passing "ok" solution gets exactly the failures' advantage. Plain GRPO on the same rewards gives [0.536, 0.391, −0.464, −0.464].
- bt→gar: advantages **[0, 0, 0, 0]**. All BT signal is erased.
- tournament→gar: only the single round-robin winner passes.

There are also two different operators named GAR. `group_ops.gar` uses λ = ΣA/ΣfA with a cap. `credit/gar.gar` (the `gar_group_fn` path) interpolates toward mass·f/Σf with `lam`. Doc §1 describes them as one operator.

- **Fix:** have GAR read the pass/fail outcome from the composed record, not from the rewritten reward. GRS keeps `grs/test`; for BT and tournament, keep the pre-stage reward in `reward_components["raw"]`. Use that outcome for the pass set, and use the GRS/BT score as the quality `f` when no `quality` is given. Alternatively, refuse a GAR whose input rewards are not in {0} ∪ [threshold, ∞). Keep one GAR definition and document which one.
- **Regression test:** `compose(grs(rub), gar(None))` on [pass (1, 1), pass (0.9, 0.95), fail, fail]. Assert both passing members get positive advantage and the failures keep negative advantage. It fails now (−0.25).

### V2 (P0): one non-finite behaviour log-prob NaNs the model

The gated corrections use `torch.where(gate, r·A·logp, 0)` (`dis` `:133`, `mimo4` `:140`, `icepop` `:122`). The backward of `where` gives the rejected branch a zero gradient, which is then multiplied by `r`. When `r = exp(logp − rollout_logp)` overflows (rollout log-prob −inf, NaN, or below about −88 in float32), 0·inf = NaN.

`r05`, 1,000 tokens with one bad behaviour log-prob: dis, mimo4 and icepop each give 1/1000 NaN gradients on that token, in both float64 (−inf) and float32 (−95). tis is safe (clamp); mispo is safe (the whole trajectory is rejected). End to end with the default trainer (`dis`): 8 rollouts with one −inf token give `valid_for_training() == (True, 'ok')`, `grad_norm = nan` and **568/568 parameters NaN** after the step. Nothing in `trainer.py`, `runtime.py` or `monitors.py` checks finiteness. The same mechanism carries a NaN behaviour log-prob.

- **Fix:** compute the weight first and zero it outside the gate (`w = torch.where(gate, r, 0)`, then `obj = w·A·logp`), or clamp `log r` before `exp`. Mask non-finite `rollout_logp` and `old_logp` tokens (count them in a metric). `Turn.missing()` should reject non-finite log-probs. The trainer should skip `opt.step()` and report it when the grad norm is not finite.
- **Regression test:** `correct(dis)` on [1, 1000] tokens with one −inf rollout log-prob: assert `torch.isfinite(logp.grad).all()`. Plus `TorchTrainer.step` on such a record: assert the parameters stay finite and the token is counted as masked.

### V3 (P1): trainer temperature is unchecked

`RuntimeConfig.temperature` stamps `Turn.sampling["temperature"]` (`runtime.py:503`). `TrainerConfig.temperature` (default 1.0) is the only value the trainer uses, and nothing compares the two. `r02`, 16 rollouts sampled at T = 0.6:

- trainer at 0.6: `rollout_probs_diff = 0.0000`, r_dis p1/p50/p99 = 1/1/1;
- trainer at 1.0: `rollout_probs_diff = 0.0253`, r_dis p1/p50/p99 = **0.576/0.921/2.044**, no error.

The suite's own top-p test has to set `TrainerConfig(temperature=0.9)` by hand.

- **Fix:** in `_items`, refuse a turn whose `sampling["temperature"]` differs from `cfg.temperature` (or use the per-turn temperature in `_logprobs`). `Runtime` should check `trainer.cfg.temperature == cfg.temperature` at construction.
- **Regression test:** a record with `sampling={"temperature": 0.6}` and a trainer at 1.0 raises `ValueError`.

### V4 (P1): sequence-level aggregations depend on the item path

`_items` builds one segment item per (rollout, actor, segment) when `_segment_ok` holds. When any turn has `top_p < 1` or R3 is on, it builds one item per **call**. `aggregate` then averages over items (`seq_mean_token_mean`, `seq_ratio`) or uses G = number of items (`const_max_len`). The doc's definition is (1/G)Σ_i over rollouts o_i.

`r01`: the same two rollouts (one with 3 calls), with top_p 1.0 versus 0.9999 and candidate sets equal to the whole vocabulary, so renormalization is the identity:

| agg | items | loss |
|---|---|---|
| `token_mean_global` | 2 / 4 | equal (Δgrad 1.9e-9) |
| `seq_mean_token_mean` | 2 / 4 | **0.000487 vs 0.166857** (Δgrad 1.7e-2) |
| `const_max_len` | 2 / 4 | **0.000487 vs 0.000244** |

Compaction segments and multi-actor rollouts also count as several "sequences" in segment mode. `test_segment_items_equal_call_items` runs only the default aggregation, so it misses this.

- **Fix:** carry a rollout index on `_Item`. Have `aggregate` average per rollout (sum the trained tokens of all items of a rollout; G = number of rollouts, max over rollout lengths). If segments as samples are wanted for CompactionRL, make that an explicit knob.
- **Regression test:** parametrize `test_segment_items_equal_call_items` over every `LOSS_AGG`, plus a case with `top_p<1` and full candidate sets. Assert equal loss and gradients.

### V5 (P1): QDQ with an HF policy leaves the trainer off the grid

`QDQ.__call__` builds `params = dict(self.model.named_parameters())`. For `HFPolicy` that gives `model.transformer.h.0…`, but `TorchTrainer.state_dict()` returns HF names (`HFPolicy.state_dict` strips the wrapper). `if name in params` is never true. `r04` (tiny GPT-2):

- engine copy on the grid: True;
- trainer weight on the grid: **False**; trainer == engine: **False** (max |Δ| 0.0154);
- with `QDQ(model=pol.model)`: equal.

The suite's test uses `TinyPolicy`, whose names coincide.

- **Fix:** resolve the trainer parameters through the same `state_dict()` naming (e.g. `model.state_dict(keep_vars=True)`), or strip a known wrapper prefix. Raise if `model` is given and any applied name was not written back.
- **Regression test:** `QDQ(model=HFPolicy(tiny))` applied to `TorchTrainer(HFPolicy).state_dict()`: assert every applied parameter in the trainer equals the published tensor.

### V6 (P1): vision records train text-only

`_plain_logprobs` calls `batch_token_logprobs(pairs, T, candidates=…)` with no `media`. `VisionHFPolicy.batch_token_logprobs` then takes the text-only path. `vision.py` says the trainer side rebuilds pixels from refs, but only `record_logprobs` does, and nothing calls it on the training path. `r08` uses the suite's own two-turn VL episode (its engine log-probs are exact to 9.5e-7 through `record_logprobs`). Through `TorchTrainer`, the trainer's log-probs differ from the engine's by **1.67e-2 and 7.36e-3** (max |Δ|) on the two calls, and the step runs.

- **Fix:** carry `prompt_media(r)[call_id]` on `_Item` and pass `media=` to `token_logprobs` / `batch_token_logprobs` (segment items: the media of the segment's chain). Refuse records with `Turn.media` when the policy cannot take media.
- **Regression test:** `TorchTrainer(VisionHFPolicy).step([vision_record])`. Assert the trainer log-probs equal `record_logprobs` within 1e-5, and that a plain `HFPolicy` refuses the record.

### V7 (P1): OPD and KL reference use the wrong distribution

`TorchTeacher.temperature` defaults to 1.0 and scores the full vocabulary. `VLLMTeacher` asks for `prompt_logprobs` at temperature 0. The behaviour log-prob is taken at the sampling temperature and renormalized over the nucleus. `r10`, with the teacher and reference **identical** to the policy (the signal should be exactly 0):

| sampling | OPD advantage mean (min, max) | k3 KL mean |
|---|---|---|
| T = 1, top_p = 1 | 0 | 0 |
| T = 0.6 | +0.017 (−0.513, +0.707) | 0.049 |
| top_p = 0.8 | **−0.195** (every token negative) | 0.018 |

With top_p < 1 the OPD term is β·log(nucleus mass) < 0 on every sampled token. It pushes the policy down regardless of the teacher. `test_teacher_equal_to_the_policy_gives_zero_gradient` covers only T = 1 and top_p = 1.

- **Fix:** score teachers and the reference at the turn's sampling temperature, renormalized over `Turn.candidates` when top_p < 1, exactly as the trainer evaluates π_θ. Alternatively, compare teacher and student on the same (un-truncated) footing: use π_θ recomputed at T rather than the truncated behaviour log-prob.
- **Regression test:** extend the teacher == policy test to (T = 0.6) and (top_p = 0.8). Assert the OPD advantage is 0 and k3 is 0.

### V8 (P1): teacher vocabulary gate is opt-in

`runtime.py:201` checks `teacher_pool.check_vocab` only when `cfg.student_vocab_id is not None`. `r14`: `RuntimeConfig().student_vocab_id = None`. A 64-token teacher declaring `vocab_id="other"` distils a 32-token student for 2 steps (16 records scored, `teacher_id='foreign'`). The vocab ids are also free strings that the caller declares.

- **Fix:** with a `teacher_pool`, require `student_vocab_id` (or derive it from the trainer's tokenizer) and refuse otherwise.
- **Regression test:** `Runtime(..., teacher_pool=pool)` without `student_vocab_id` raises.

### V9 (P1): speed bonus on failures

`_shaped_rewards` gates on `b > 0` (`:319` baseline, `:328` bonus). `r13`, group [pass 10 s, pass 10 s, fail with κΦ = 0.18 at 1 s, fail 0 at 1 s], `speed_coef = 0.5`: shaped rewards **[1.0, 1.0, 0.63, 0.0]**. The partial-credit failure gets +0.45, and the passing members get 0 because the failure pulls the median.

- **Fix:** add `pass_threshold` (default 1.0) and gate with `b >= pass_threshold`, as `shaping.with_speed` and GAR already do.
- **Regression test:** the `r13` group: assert the failure gets no `shaping/speed` and its shaped reward stays 0.18.

### V10 (P1): vLLM nucleus rebuilt too small

`_nuclei` sorts the top-k entries and stops at cumulative mass ≥ top_p. vLLM's `processed_logprobs` include top-p, so the nucleus members' probabilities sum to 1, which reaches `top_p` early whenever C_{n−1} ≥ top_p·C_n. `r16`: true post-temperature probabilities [0.50, 0.36, 0.09, 0.05], top_p = 0.9, true nucleus {0, 1, 2}:

- sampled token 0: rebuilt nucleus **{0, 1}**, so the trainer's log-prob is off by **+0.0995 nats** on that token;
- sampled token 2: the call fails ("outside the rebuilt nucleus").

The suite's mock feeds pre-top-p probabilities, which vLLM in this mode does not return. This was not run against a real vLLM (no GPU); the semantics come from vLLM's `--logprobs-mode` definition ("processed: after … temperature and top_k/top_p").

- **Fix:** with processed log-probs, the nucleus is the set of finite entries. Require that set to hold all the finite mass (k ≥ nucleus size, i.e. the finite entries sum to 1 ± ε) and that the sampled token is in it.
- **Regression test:** the `r16` input. Assert `_nuclei` returns {0, 1, 2} for both sampled tokens.

### V11 (P2): advantage sources dropped without a word

`r09`:

- (a) `credit_unit="reward_to_go"` with GAR advantages [0.9, 0.1, −0.5, −0.5] gives first-token advantages [0.5, 0.5, −0.5, −0.5]. `per_token` overrides `advs` on both item paths. `estimator` and `std_normalize` are also ignored there.
- (b) Critic mode: `latency_coef = 5` leaves GAE unchanged, because `_gae_advantages` reads `r.reward`, not `_shaped_rewards`.
- (c) Critic mode with `format_penalty_mode="signed"`: the malformed call's tokens are untouched, because the sign comes from `advs`, which is all 0 in critic mode. `reduction` does apply (−1.0).

- **Fix:** refuse the combinations, or compose them. For rtg, add the GAR delta per rollout. Feed shaped rewards into GAE. Take the sign for `signed` from the token advantages (or from the rollout's GAE sum).
- **Regression test:** each combination either raises or changes the advantages as specified.

### V12 (P2): truncation bootstrap off by one token

`token_values` returns V(s_0 … s_{T−1}), the state before each completion token. The trainer passes `bootstrap_value=float(v[-1])`, which is V(s_{T−1}). `r07`, with V(s_T) computed by one more value position: trainer advantages [0.0352, −0.1484, −0.0736, **0.0**], reference [−0.0070, −0.1906, −0.1158, −0.0422]. With γ = 1 the last TD error is always r + V(s_{T−1}) − V(s_{T−1}) = r. `test_truncation_bootstraps_from_the_final_value` sets V ≡ 0.5, so it cannot tell the two positions apart.

- **Fix:** have the critic return T+1 values (append the hidden state after the last token) and bootstrap from the last one.
- **Regression test:** a non-constant critic: assert the last advantage equals γ·V(s_T) − V(s_{T−1}) for a truncated rollout with reward 0.

### V13 (P2): token budget ignored for sequence aggregations

`single = loss_agg != "token_mean_global"` makes `mbs = [mine]`. `r15`, budget 20, 288 forward tokens: 16 micro-batches for `token_mean_global` and **1** for each of the other four. Each of those aggregations is a sum of per-item terms with a known global normalizer (per-item 1/(G·|o_i|), per-prompt 1/(P·tokens_p), 1/(G·L)), so they decompose exactly.

- **Fix:** compute per-item weights from the global batch, then micro-batch with `J = Σ w_i Σ_t J_it`.
- **Regression test:** each `LOSS_AGG` at budget 20 versus 10⁶: equal gradients, more than one micro-batch.

### V14 (P2): HFPolicy skips post-head logit transforms

`_head_rows` is `linear(h, W)/T`. `r12`: Qwen3 max |Δ| = 4.8e-7 (fine); Gemma-2 (softcap 0.5) **6.9e-3**; Granite (`logits_scaling = 8`) **1.4e-1** versus `log_softmax(model(...).logits)`.

- **Fix:** apply the model's `final_logit_softcapping`, `logits_scaling` and `logit_scale` in `_head_rows`, or refuse configs that set them.
- **Regression test:** tiny Gemma-2 and Granite: `HFPolicy.token_logprobs` equals the model-forward log-probs within 1e-5.

### V15 (P2): MXFP4 midpoint rounding

`grid[argmin |v − grid|]` returns the first (lower) index on ties. `r11`: 0.75→0.5, 1.75→1.5, 3.5→3.0, where round-half-to-even gives 1.0, 2.0, 4.0. The published values are self-consistent, since re-QDQ is idempotent. But they differ from what an MXFP4 kernel or converter computes from the master weights.

- **Fix:** break ties toward the even E2M1 code.
- **Regression test:** the `r11` midpoint vector maps to [0, 1, 1, 2, 2, 4, 4].

### V16 (P2): group operators score unscored members

`r18`:

- `zero_hacks` sets the reward of an ungraded member to 0.0;
- `bt_pairwise` and `tournament` give every member a reward, including members with reward None or `mask_sample`.

`_run_group_fn` passes every `finished` member with a value, and M10-masked World records arrive that way (reward None, `mask_sample` True). The trainer still skips `mask_sample`. But a record with reward None and `mask_sample` False becomes scored: the GRPO advantage moves from [0.5, −0.5, 0] to [0.667, −0.333, −0.333]. The tournament also pays the judge for masked members, and their games enter the real members' win shares.

- **Fix:** every operator acts on `scored = [m for m in members if m.reward is not None and not m.mask_sample]` only, as `gar` already does.
- **Regression test:** each operator on [pass, fail, masked]: the masked member keeps reward None, and the tournament denominator is the number of scored members minus 1.

## 3. Tests (V17, P1)

- **Vacuous for what they claim:**
  - `test_truncation_bootstraps_from_the_final_value` (constant V hides V12);
  - `test_trainer_advantages_equal_direct_gae_including_segments` (it rebuilds the trainer's own `gae` call: wiring only);
  - `test_segment_items_equal_call_items` (default aggregation only: misses V4);
  - `test_qdq_makes_trainer_and_engine_bit_identical` (TinyPolicy only: misses V5);
  - `test_teacher_equal_to_the_policy_gives_zero_gradient` (T = 1, top_p = 1 only: misses V7);
  - `test_top_p_candidates_rebuilt_from_top_k_logprobs` (it feeds pre-top-p probabilities as "processed": misses V10);
  - the vision tests never go through a trainer (V6).
- **Never exercised:**
  - PPO `dual_clip`;
  - GSPO's gradient and its clipping (only r = 1 is tested);
  - `prompt_mean`, `seq_ratio` and `const_max_len` (doc §7 says "golden tests for every correction" and checklist #18 says "loss_agg + DAPO golden: done");
  - any non-finite input to `correct`;
  - temperature or top-p mismatch between record and trainer.

  `r17` shows the code is right for dual clip, GSPO, `prompt_mean` and `const_max_len`, so these are coverage gaps.
- **Sketch:** add `r17`'s four checks as goldens; turn each regression test above into a unit test; add a property test `TorchTrainer.step` → `isfinite(params)` under random ±inf/NaN behaviour log-probs.

## 4. Probe index

| probe | finding |
|---|---|
| `r01_seqmean_item_unit.py` | V4 |
| `r02_temperature.py` | V3 |
| `r03_grs_then_gar.py` | V1 |
| `r04_qdq_hfpolicy.py` | V5 |
| `r05_gate_nan.py` | V2 |
| `r06_microbatch_equivalence.py` | §0 (no defect) |
| `r07_truncation_bootstrap.py` | V12 |
| `r08_vision_trainer_drops_images.py` | V6 |
| `r09_advantage_sources_dropped.py` | V11 |
| `r10_opd_temperature_topp.py` | V7 |
| `r11_qdq_grid.py` | V15, §0 |
| `r12_hfpolicy_head_bypass.py` | V14 |
| `r13_speed_gate_partial_credit.py` | V9 |
| `r14_teacher_vocab_unchecked.py` | V8 |
| `r15_token_budget_ignored.py` | V13 |
| `r16_vllm_nucleus_rebuild.py` | V10 |
| `r17_untested_branches_check.py` | V17, §0 |
| `r18_group_ops_unscored_members.py` | V16 |
