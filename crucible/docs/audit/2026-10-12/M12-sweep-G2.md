# M12 framework sweep, group 2: MiMo's verl fork and verl, at code level

**Scope.** Crucible M12 `train` (HEAD `3b522cb`) against two training stacks:

| Short | Source | Pin | Licence | How it was run |
|---|---|---|---|---|
| `V` | `RL-Environments/06-frameworks-and-env-code/infra-volcengine-verl/` | `5eb4826` | Apache-2.0 | `verl/trainer/ppo/core_algos.py` and `rollout_corr_helper.py` executed for real, with stubs for omegaconf, tensordict, `verl.utils.device`, `verl.protocol` (`t1_loader.py`, `t3_verl_loader.py`) |
| `XV` | `RL-Environments/06-frameworks-and-env-code/xiaomimimo-verl/` | `a2ad9f6` | Apache-2.0 (`XV/LICENSE`; `third_party/mimoagent-osr` MIT) | same loader; read for the v1 trainers |
| `ME` | `RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss/` | HF release | apache-2.0 (front matter) | read |
| `SL` | `RL-Environments/06-frameworks-and-env-code/infra-thudm-slime/` | as pinned in `00-source-pins.txt` | Apache-2.0 | read (R3 row contract only) |

`C` = `crucible/src/crucible/`, `T` = `crucible/tests/unit/`. Probes: `/private/tmp/claude-501/-Users-muahmed-Desktop-Projects-RL-for-Real-World-Environments/54b8300a-e533-4d3f-9f31-c0f8369fa981/scratchpad/m12/g2/` (`t1_*` algorithms, `t2_*` control flow, `t3_*`/`t4_*` bridge and scale features; each has a `.out`). `[ran]` = executed with `crucible/.venv/bin/python` (torch 2.14), CPU, float64 unless stated.

**What the MiMo fork does not contain.** `XV` is verl plus MiMo recipes. It has no code for the four-bound mask, entropy-adaptive bounds, GAR, GRS, MXFP4 QDQ of trainer weights or top-p candidate capture (grep over `XV/verl`, `XV/recipes`, `ME`: no hits). Those Crucible features have only the article (`docs/research/m11-m12/06-article-systems.md:587-596, 442, 467`) as reference. For them this sweep checks the math against the closest verl mechanism and against the article's formula. `XV`'s public recipe is GRPO without std, PPO clip 0.2, `prompt-mean`, `invalid_reward_value: -999`, `filter_groups`, `length_penalty`, top_p 0.95 (`XV/recipes/general/config/general.yaml:41-99`).

## 0. Headline defects

| # | Sev | Defect | Crucible | Reference | Evidence |
|---|---|---|---|---|---|
| H1 | **P0** | The documented group chain `zero_hacks → grs → gar` zeroes every advantage of a mixed group. `group_ops.gar` binarizes rewards at `pass_threshold = 1.0`, and GRS has just scaled every passing reward below 1. The trainer then trains on the zeros, because every scored member carries an advantage | `C/train/group_ops.py:5-6` (chain), `:92` (binarize), `:112-113` (sets A); `C/train/trainer.py:408-416` | article §3.6.3.2 (GRS then GAR); `C/credit/gar.py:51` keeps the real rewards | [ran `t1_gar`] rewards [0.9, 0.48, 0, 0] → advantages **[0, 0, 0, 0]**; GRS alone gives [0.555, 0.135, −0.345, −0.345] |
| H2 | **P0** | `QDQ(model=...)` writes the MXFP4 grid into the trainer's fp32 master weights at every publish. An optimizer step smaller than half a grid step is rounded away, so QDQ'd parameters stop learning. QAT keeps an unquantized master and fake-quantizes the forward | `C/train/weights.py:150-158`; asserted by `T/test_infra_algorithms.py:157-171` | MiMo Fig. 6 "QDQ weights" edge goes trainer → inference only (`06-article-systems.md:442`); verl's refit quantizes a copy (`XV/verl/utils/vllm/vllm_fp4_utils.py`) | [ran `t1_qdq`] AdamW lr 2e-6 (MiMo's lr), 50 steps with a publish each: **49 of 28,672** QDQ'd elements ever left their first grid value |
| H3 | **P0** | vLLM top-p candidate sets are rebuilt wrongly. Crucible requires `--logprobs-mode processed_logprobs` (`vllm_http.py:39`), which vLLM computes *after* top-p: renormalized over the nucleus, −inf outside (measured on vLLM 0.25.1, `meta-pytorch-openenv/rfcs/006-agentic-rl-harness-interception.md:241`). `_nuclei` then takes "the smallest prefix whose mass reaches top_p" of these renormalized probabilities, which cuts the nucleus short. With unprocessed logprobs the behaviour logprob is unrenormalized instead. Either way the trainer's renormalized π_θ differs from π_rollout at θ = θ_rollout | `C/train/vllm_http.py:136-158` | vLLM processed mode per the RFC; `C/train/sampling.py:21-32` is correct (0/4000 differ from HF `TopPLogitsWarper`) | [ran `t1_nucleus`] top-k = V: wrong set in **3340/4000** calls with no error, 479 raise; trainer − engine logprob at equal weights: mean **0.13**, max 0.35 nats. Test fixture feeds unrenormalized probs (`T/test_vllm_engine.py:132-142`) |
| H4 | P1 | Sequence-level terms silently switch unit from trajectory to model call when any call sampled with top_p < 1 or under R3 (`_segment_ok`). Then `seq_mean_token_mean`, GSPO's sequence ratio and the MIS-PO / HisPO trajectory gates act per call. verl and MiMo keep one row per trajectory | `C/train/trainer.py:270-274, 460-481` | `XV/recipes/general/token_trace.py:73-97` (one masked row per trajectory) | [ran `t1_item_unit`] same 2 rollouts, top_p 1.0 → 0.95 with a full-vocabulary nucleus (π_θ unchanged): items 2 → 4, `seq_mean_token_mean` grad-norm 0.102 → **0.139**, GSPO loss 0.1663 → 0.1649 |
| H5 | P1 | R3 over SGLang cannot work. SGLang returns `routed_experts` for every token but the last (prompt + output − 1 rows; continuations request `routed_experts_start_len`); Crucible treats each row as one output token, concatenates on resume, and flattens layers into one list that `TinyMoEPolicy` cuts to layer 0 (`engine.py:286`). The trainer's 100 % gate then refuses every call. The test fixture returns one row per output token | `C/train/sglang_http.py:80-83`; `C/train/engine.py:286`; `C/train/trainer.py:470-472`; `T/test_sglang_engine.py:26` | `SL/slime/rollout/sglang_rollout.py:191-238`; `SL/slime/backends/megatron_utils/cp_utils.py:468-476` | [ran `t1_r3_sglang`] 10-token prompt, 4 output tokens → **13** routing rows; resume → 17 rows for 6 tokens; gate fails both |
| H6 | P1 | `crucible_delta` reads the outcome only at the last *sampled* token; verl places the scalar reward at the last attended position. A response that ends on observation tokens trains on A = 0 | `C/train/bridges/verl_bridge.py:41-44` | `V/verl/experimental/agent_loop/agent_loop.py:143-148, 1110-1116`; verl grpo sums the row | [ran `t3_bridge_probe` E] [0, 0, 0, 0] vs verl [0.5, −0.5, 0.5, −0.5] |
| H7 | P1 | The verl bridge would not run: the adapter dict is not a valid `AgentLoopOutput` (`metrics` is required), no `AgentLoopBase` exists, and turn credit has no channel into verl (`token_rewards()` has no caller; `extra_fields` never reach an estimator). `M12-train.md:142` claims the turn-credit path | `C/train/bridges/adapters.py:22-26`; `verl_bridge.py:23-33` | `V/.../agent_loop.py:88-115, 227, 487`; `V/verl/trainer/ppo/ray_trainer.py:249-279` | [ran `t3_bridge_probe` F] `ValidationError: metrics`; `rm_scores = [0, 0, 0, 0, 0, 1.0]` |
| H8 | P1 | MiMo's default aggregation (`prompt-mean`) and every non-token-mean mode run the whole global batch as one micro batch (token budget ignored) and are refused under DP. MiMo's per-row weights split exactly | `C/train/trainer.py:606-607, 625, 700-701` | `XV/verl/trainer/ppo/core_algos.py:1160-1182, 1217-1222` | [ran `t3_agg_probe` 1, 3] 1 micro batch vs 8; MiMo weights over 4 parts = Crucible full batch to 2.8e-17 |
| H9 | P1 | The trainer never applies `valid_for_training`, and the M12 rollout paths build records in eval mode, so `capture_incomplete` is never masked. A call with `logprobs=None` trains with behaviour logprob 0.0 (π_b = 1). `M12-train.md` §7 says "train mode masks capture_incomplete" | `C/train/world_env.py:189` and `C/interop/nemo_gym/client.py:155` call `build` without `mode="train"`; `C/runtime/record.py:102`; `C/train/bridges/common.py:79`; no `valid_for_training` call under `C/train/` except its definition (`control_record.py:107`) | MiMo `TokenTrace` (`XV/recipes/general/token_trace.py:73-97`) | [ran `t2_incomplete_capture`] record reports `(False, 'token record incomplete ... logprobs')`; item behaviour logprobs `[-1.0, -1.2, 0.0, 0.0, 0.0]`; the step trains 2 items / 8 tokens |
| H10 | P2 | τ = 4 is labelled "MiMo's published value" but is one version looser than MiMo's code: MiMo's threshold counts versions *spanned* (span = g − s + 1), so T = 4 bounds the lag at 3 | `C/train/scheduler.py:47` | `XV/verl/trainer/ppo/v1/replay_buffer.py:632-641, 660-667` | [ran `t2_mimo_rb`, `t2_staleness`] MiMo's own `ReplayBufferAsync`, T = 4: max trained lag 3 (drop and wait, 5 seeds); Crucible τ = 4: 4 |

The core per-token math is right. Every correction with a verl counterpart (PPO with dual clip, TIS, IcePop, CISPO, GSPO, DIS, the four-bound mask, MIS-PO) gives the same loss and gradient as verl to ≤ 1e-16 under token-mean and MiMo prompt-mean, and to ≤ 6e-9 under seq-mean-token-mean (verl's 1e-8 epsilon). The token-mean normaliser is global across micro batches and DP ranks, exactly as verl's `global_batch_info` (`t3_agg_probe` 5: diff 0.0). The defects are in what surrounds that math: group operators, weight transforms, engine adapters, the training unit, and the bridge.

## 1. Numerical equivalence (task 1)

### 1.1 Corrections × aggregation [ran `t1_corrections`]

Tensors: B = 8 rows, T = 12, per-row advantages [1.5, −2, −1, 0.7, −0.3, 2, −5, 0], lengths [12, 7, 12, 3, 9, 1, **0**, 5] (row 6 all-masked), prompt ids [0,0,1,1,2,2,3,3]. "normal": |log r| ≲ 1; "extreme": log-ratios ±3 and **±25** (past verl's ±20 clamp) on chosen tokens. Crucible loss = −`aggregate(correct(...).objective)` (`C/train/mismatch.py:89-183`, `C/train/objectives.py:37-62`); gradient = d loss / d logp by autograd on both sides.

| Crucible | verl / MiMo-verl reference | token-mean | seq-mean-token-mean | prompt-mean (`XV` weights) | const_max_len vs seq-mean-token-sum-norm(L = T) |
|---|---|---|---|---|---|
| `ppo` ε .2/.28, `dual_clip=3` | `compute_policy_loss_vanilla`, `clip_ratio_c = 3` (`V/core_algos.py:1285-1376`) | 0 / 0 | 2e-9 / 2e-9 | 6e-17 / 0 | **7/8 ratio** (see below) |
| `ppo`, no `dual_clip` (Crucible default) | vanilla (dual clip always on, default c = 3, `:1323`) | normal 0; **extreme 1.47e9** | extreme 8.6e8 | extreme 1.2e9 | — |
| `tis` C = 2 | vanilla × `compute_rollout_correction_weights(token, 2.0)` (`V/rollout_corr_helper.py:522-596`) | 0 / 0 | 2e-9 | 0 | 7/8 |
| `icepop` [0.5, 5] | vanilla × IS weights `"0.5_5.0"` (IcePop branch, `:597-603`) | 0 / 0 | 2e-9 | 0 | 7/8 |
| `cispo` .2/.28 | `compute_policy_loss_cispo` (`:2047-2105`) | 0 / 0 | 5e-9 | 1e-16 | 7/8 |
| `gspo` 3e-4/4e-4 | `compute_policy_loss_gspo` (`:1545-1619`) | 1e-16 | 2e-9 | (`XV` gspo forces seq-mean-token-mean: `XV/core_algos.py:1657`) | 7/8 |
| `dis` (0.8, 3.0) | `bypass_mode`, `loss_type=reinforce`, `rollout_is="token"`, threshold `"0.2_4.0"` (`:2420-2559`, `:2392`) | 4e-17 / 7e-18 | 5e-9 | 1e-16 | 7/8 |
| `mimo4` [.2, 5] both signs | same, `"0.2_5.0"` | 4e-17 / 7e-18 | 5e-9 | 1e-16 | 7/8 |
| `mispo` tok [.5, 2], traj band | `compute_rollout_rejection_mask("token_k1,seq_mean_k1")` + `compute_policy_loss_reinforce` | 0 when the RS mask is applied as a weight; **2.4e-3 / 8.8e-2** (normal / extreme) when verl's RS mask also shrinks the denominator | 7e-9 vs 2.4e-2 | 0 | — |

Values are max |Δloss| / max |Δgrad|; one number = both. Semantic differences, each legitimate unless marked:

1. **Dual clip default.** Crucible's `ppo`/`tis`/`icepop`/`kpop` apply dual clip only when `params["dual_clip"]` is set (`mismatch.py:112-113`); verl always applies c = 3 and also clamps log r to ±20 (`V/core_algos.py:1338`). With dual clip on, the ±25 tokens give identical results (the clamp never matters). With Crucible's default, a negative-advantage token at r = e^25 gives a loss of 1.5e9. Reachable whenever r_s ≠ 1: `updates_per_batch > 1`, or `old_logprob_source="rollout"` under staleness. **P2**: default dual clip 3 for the PPO family.
2. **const_max_len vs seq-mean-token-sum-norm.** Two differences. Crucible divides by all rows `J.shape[0]` (`objectives.py:61`), verl by non-empty rows (`V/core_algos.py:1183`); the probe's all-masked row gives exactly 7/8. The trainer never builds an all-masked row (segment items need `sum(mask) > 0`, `trainer.py:288`), so this stays latent. The live one: the trainer never passes `max_len` (`trainer.py:638`), so L is the batch's longest sequence and changes every step, while HisPO and verl's `loss_scale_factor` are constants (`V/verl/workers/config/actor.py:164`). **P2**.
3. **MIS-PO denominators.** verl's rollout rejection (`rollout_rs`) rewrites `response_mask`, so rejected tokens leave the token count (`rollout_corr_helper.py:1048-1061`); Crucible keeps them in the denominator (`trainer.py:603` counts `mask`, not `kept`). Crucible matches the MIS-PO formula L = −E[𝕀·𝕀·log π·Â] (expectation over all tokens) and verl's bypass path; verl's decoupled RS path renormalizes over the survivors. Legitimate; document the choice.
4. **verl's k1 bands are inverted.** `token_k1 = −log r` (`V/rollout_corr_helper.py:263`) and the check is `log(lower) ≤ −log r ≤ log(upper)` (`:316`), so a `"lo_hi"` spec keeps r ∈ [1/hi, 1/lo]. With MIS-PO's asymmetric trajectory band (−0.004 ≤ mean ln x ≤ 0.001) a literal port to verl keeps the mirror band [ran: asym case, loss −0.507 vs +0.248]. Crucible applies the band to r as the article states (`mismatch.py:142-146`). Not a Crucible defect; any bridge config must invert.
5. **Four-bound mask** has no sign split in verl; with equal bands it equals verl bypass-reinforce IcePop. Crucible's inclusive [0.2, 5] equals verl's inclusive IcePop; `dis` uses open bounds (0.2, 4.0) (`mismatch.py:131`), which differ only on exact boundary ratios.
6. **GSPO** is identical but for verl's `clamp(log s, max=10)` (`V/core_algos.py:1590`): no token in the probe reached it. Crucible has no clamp. P2.

No reference: `kpop`, `hispo` (neither verl nor `XV` implements them); entropy-adaptive bands (article only). Crucible's band update follows the article's direction (`trainer.py:375-378`: entropy low → positive band wider, negative tighter), but its entropy is −mean behaviour logprob (`trainer.py:368-371`), a one-sample estimate of π_rollout's entropy and biased low under top-p truncation (see §4, G2-T4g).

### 1.2 KL, GAE, value loss [ran `t1_kl_gae`]

| Item | Crucible | verl | Result |
|---|---|---|---|
| k3 KL to the reference | w·(e^(ref−lp) − (ref−lp) − 1), unclamped (`trainer.py:534-535`) | `kl_penalty("low_var_kl")`: clamp d to ±20, result to ±10 (`V/core_algos.py:2242-2245`) | normal: **0 / 0**; extreme (ref − lp = 25): value **7.2e10**, gradient 7.2e10 vs clamped 10 / 0. Crucible also lacks verl's `k3+` straight-through (unbiased KL gradient): grad diff 0.44 on normal data. **P2** |
| GAE | `objectives.gae` (`objectives.py:148-207`), skip-observation | `compute_gae_advantage_return` (`V/core_algos.py:216-262`) | raw advantages (verl returns − values) equal to **2.8e-8 / 5.3e-8** (Crucible returns float32) with and without observation gaps. Two differences: verl **whitens** the advantages it returns (`:262`), Crucible does not; a reward that lands on an observation position is carried to the next trainable token by Crucible and **dropped** by verl (Δ 0.5). Crucible's carry is the better choice |
| Value loss | MSE, global token mean, no clip, no ½ (`trainer.py:249-268`) | ½·max((v−R)², (v_clip−R)²) (`V/core_algos.py:2125-2186`) | scale differs by 2 (absorbed by lr); no value clipping in Crucible. P2 |

### 1.3 Advantages [ran `t3_bridge_probe` I, 8 groups × 4]

| Crucible `group_advantages` (`objectives.py:68-110`) | verl estimator | max |Δ| | Semantics |
|---|---|---|---|
| `mean` | `grpo`, `norm_adv_by_std_in_grpo=False` | 0.0 | n = 1: verl A = R, Crucible A = 0 |
| `mean` + `std_normalize` | `grpo` (Bessel `torch.std`) | 0.219 | Crucible uses population std (`:107`): A is √(n/(n−1)) = 1.155× verl's at n = 4. P2: name the convention |
| `rloo` | `rloo` | 0.0 | identical |
| `remax` | `remax` | 0.0 | identical for outcome rewards |
| `reinforce_pp` | `reinforce_plus_plus` | 0.093 | verl whitens over **tokens**, Crucible over rollouts; verl's `reinforce_plus_plus_baseline` is missing. P2 |

Masked rows: Crucible drops them from every statistic and gives A = 0 (`objectives.py:77, 90-93`). `XV` fills an invalid row with the group's valid mean, then sets A = 0 (`XV/core_algos.py:304-350`). With `std_normalize` the filled rows shrink the std (they sit on the mean and count in n), so the two differ by construction. Crucible's handling is cleaner. §2 covers the loss-denominator side.

### 1.4 GAR and GRS [ran `t1_gar`]

Crucible has **two** GAR implementations:

| | `C/train/group_ops.py:83-116` `gar` | `C/credit/gar.py:34-64` `gar` / `gar_group_fn` |
|---|---|---|
| Rule | A' = λ·f·A on passing members, λ = Σ_P A / Σ_P f·A, capped at `lambda_max`, re-centred when capped | A' = A + λ·(M·f/Σf − A), λ ∈ [0, 1] (default 0.5) |
| Statistic | on **binarized** rewards (`r ≥ pass_threshold` → 1, else 0, `:92`) | on the real rewards |
| Binary group [1, 1, 0, 0], f = [1, .5] | [0.667, 0.333, −0.5, −0.5] | λ = 0.5: [0.583, 0.417, −0.5, −0.5]; λ = 1: identical to `group_ops` (0.0) |
| After GRS (rewards 0.9, 0.48) | **[0, 0, 0, 0]** (H1) | [0.555, 0.135, −0.345, −0.345] (no passing member at threshold 1: plain mean centring, signal kept) |
| Partial credit 0.5 | → treated as 0: [0.75, −0.25, −0.25, −0.25] vs group mean [0.625, 0.125, −0.375, −0.375] | kept |
| Cap binds (`lambda_max = 1.2`) | failing members move −0.5 → −0.475 (docstring: "failing rollouts untouched") | n/a |

MiMo's article rule (survey `docs/research/2026-10-09-agentic-rl-infra-survey.md:101`): zero confirmed hacks, recompute group statistics, redistribute positive mass among passing solutions by quality with total positive mass conserved. Both implementations conserve mass when uncapped; `group_ops` loses the GRS signal (H1) and partial credit. Test coverage: `T/test_infra_algorithms.py:104-139` composes `zero_hacks → grs` and runs `gar` alone, never the documented three-stage chain. **Fix:** one GAR (keep `credit/gar.py`), and `group_ops.gar` computes the statistic on the rewards it receives with a `passing` predicate that GRS output satisfies (e.g. `R_test > 0` recorded as `grs/test`).

The runtime fallback (late or raising operator → raw advantages, `group/fallback`) matches the article's "falls back to the original advantage" (§2 has the control-flow details).

### 1.5 MXFP4 QDQ [ran `t1_qdq`]

| Check | Result |
|---|---|
| Scale and grid vs OCP MX v1.0 (E8M0 X = 2^(⌊log2 amax⌋ − 2), E2M1, saturate ±6) | fp32 weights: **0 %** of elements differ |
| Rounding ties | Crucible's `argmin` takes the lower grid point (`weights.py:121`): 0.75 → 0.5, 1.75 → 1.5, 3.5 → 3.0. OCP / hardware converters round to nearest even: 1.0, 2.0, 4.0. On **bf16** weights ties are common: **0.47 %** of elements differ (max 1.6e-2). The trainer↔engine bit identity holds only against Crucible's own engine, not a real MXFP4 kernel. P2 |
| Default selection | `include=("*",)` QDQs every non-norm/bias/embed matrix incl. attention and `lm_head` (`weights.py:131`); MiMo quantizes expert weights. P2 (footgun) |
| Master weights (`model=`) | H2: 99.8 % of QDQ'd elements frozen at lr 2e-6 |

### 1.6 Top-p candidates [ran `t1_nucleus`]

`sampling.nucleus` equals the HF/vLLM `TopPLogitsWarper` rule on 4000 random distributions (0 differ), and `renormalized` is the exact log π − log Σ_nucleus π. The engine-side rebuild for vLLM is the defect (H3). Three runs: top-k 20, wide logits: 2414/4000 silent wrong sets; top-k = V: 3340/4000; top-k = V, sharp logits (×4): 2388/4000, mean bias 0.097 nats. **Fix:** under processed logprobs the nucleus is exactly the set of finite entries (require top-k ≥ nucleus size, detected by the presence of a −inf entry); and refuse `top_p < 1` when the server is in raw mode.

### 1.7 R3 routing replay

H5 covers the SGLang path. Two more gaps:
- **Prompt routes.** `TinyMoEPolicy.token_logprobs` replays only the completion positions; prompt positions keep the trainer's router (`engine.py:300-305`). In TinyMoE the MoE layer is last, so this is exact. In a transformer, prompt-token routes at layer l shape what completion tokens attend to at layer l + 1, and slime / verl replay every position (`SL/cp_utils.py:472`: rows = tokens − 1). A real-model R3 must take prompt routes too.
- `HFPolicy` has no routing replay and `FSDPTrainer` refuses `r3` (`C/train/fsdp.py:74`). R3 is CPU-TinyMoE only, which `M12-train.md` §9 #36 says only for vLLM.

Resume handling (`runtime.py:270`, `engine.py` `resume_routing`) keeps the recorded routes of the partial output, which is right. slime additionally refuses a continuation whose captured rows ≠ tokens − 1 (`SL/sglang_rollout.py:193-197`); Crucible has no such check.

### 1.8 Training unit [ran `t1_item_unit`]

H4. The trainer builds one item per (actor, segment) only when `_segment_ok`; otherwise one item per call with its full prompt (`trainer.py:460-481`). Token-mean is unaffected (the sum over tokens is the same). Every row-level term is not: `seq_mean_token_mean`/`seq_ratio` average over calls, `gspo` takes a per-call sequence ratio, `mispo`'s trajectory geometric mean and `hispo`'s sequence gate are per call, `const_max_len`'s row count grows. MiMo samples at top_p 0.95 (`XV/recipes/general/config/general.yaml:45`), so a MiMo-like run with GSPO or seq-mean would silently change objective. **Fix:** carry a `seq_id` (rollout, actor, segment) on each item and aggregate row-level terms over `seq_id` (the `seq_ids` argument `mismatch.py:33` already names and never uses).

## 2. MiMo `colocate_async` vs Crucible `colocated_partial` (task 2)

`XV/verl/trainer/ppo/v1/trainer_colocate_async.py` (59 lines) supplies four hooks to `trainer_base.PPOTrainer`. Read end to end with `trainer_base.py`, `replay_buffer.py`, `agent_loop_tq.py`, `verl/workers/rollout/llm_server.py` (`FullyAsyncLLMServerClient`) and `checkpoint_engine/base.py`.

### 2.1 Control flow

| Phase | MiMo (`XV`) | Crucible | Verdict |
|---|---|---|---|
| Warmup | `num_warmup_batches` (default 1) whole batches submitted before step 1 (`trainer_colocate_async.py:40-46`) | waits for B × (1 + `warmup_batches`) terminal groups (`C/train/runtime.py:691-695`) | equivalent |
| Admission | one batch prefetched per step (`trainer_base.py:777-778`); evicted groups refilled one for one (`replay_buffer.py:684-687`); no explicit in-flight cap | S·B·n untrained rollouts, in flight + buffered (`C/train/scheduler.py:106-127`) | Crucible stricter: fine |
| S1 sample | `replay_buffer.sample`, oldest submission step first (`replay_buffer.py:495-504`); `on_sample_end`: `abort_replicas` + `sleep_replicas` (weights and KV freed) (`trainer_colocate_async.py:55-59`; `checkpoint_engine/base.py:448-460`) | `select` → `abort_all` + `pause` (`C/train/runtime.py:700-709`); `oldest_first` by submit index (`scheduler.py:191-192`) | same |
| In-flight sequences | **token-level resume, always**: the client resubmits prompt + tokens so far after the abort, keeps old logprobs, pins the cumulative `max_tokens` (`llm_server.py:374-445`); environments keep running | default `version_switch="call"` (`runtime.py:68`): the partial output is discarded and the call re-issued (prompt re-prefill); `"token"` keeps it and re-prefills prompt + partial, or skips the re-prefill with `resume_keeps_kv` (`runtime.py:258-278`) | design difference: Crucible's default buys call-level version purity with wasted decode. Set `"token"` for MiMo parity (D-T2-8, P2 note) |
| S2 publish | `on_step_end`: `update_weights(global_steps)` + `resume_generation_replicas` every step (`trainer_colocate_async.py:48-53`) | every `publish_every` steps, then `engines.resume()` (`runtime.py:733-745`) | same at K = 1 |
| π_old | recomputed with pre-update weights (`trainer_base.py:1771-1838`); rollout drift uncorrected unless `algorithm.rollout_correction` is set, and the general recipe does not set it | default `dis` needs only rollout logprobs; implicit recompute refused for multi-version rollouts (`trainer.py:611-615`) | Crucible stricter: fine |
| Checkpoint resume | in-flight prompts re-issued from scratch with `global_steps` reset; finished siblings deleted (`trainer_base.py:1108-1144`) | live members restart with `versions=[]`, finished siblings kept, `oldest()` falls back to the original `submit_version` (`C/train/state.py:134-141`; `pool.py:47-55`) | Crucible does not reset the staleness clock: fine |

### 2.2 Staleness

| | MiMo | Crucible |
|---|---|---|
| Unit | per prompt group: span = g − s + 1, s = submission step (`replay_buffer.py:632-641, 660-667`) | per group: v_train − min(version that served any member's call), fallback submission version (`pool.py:47-55, 296-298`) |
| drop / wait | evict terminal spans > T / block while a pending or running prompt has span ≥ T | evict staleness > τ / block while any live group (finished members included) has staleness ≥ τ (`scheduler.py:159-171`) |
| Per-token | metrics only: per call `min/max_global_steps` (`llm_server.py:422-447`); `trajectory_staleness = (g − 1) − max_global_steps` (a lower bound) and a `_worst` variant (`trainer_base.py:2447-2475`) | `version_spans` per token; `staleness_worst/*` |
| Bound | T = 4 (`recipes/design/config/webdev.yaml:105`, `music.yaml:104`), T = 2 (`recipes/code/run_train.sh:283`) | τ = 4, "MiMo's published value" (`scheduler.py:47`) |

[ran `t2_mimo_rb`, `t2_staleness`] MiMo's own `ReplayBufferAsync` (imported verbatim, tq/omegaconf stubbed), long-tail schedule, 5 seeds: T = 4 → max trained lag 3, T = 2 → 1. Crucible's simulator at τ = 4 trains up to 4, under wait and drop. Same unit (versions behind the pre-update weights): **H10**, map τ = T − 1. Crucible's `wait` also counts finished members of open groups and has `bound_overflow`; MiMo's `wait` checks in-flight prompts only.

### 2.3 R3 across resumes

MiMo keeps the old rows and appends only `routed_experts[-len(new tokens):]` on resume (`XV/verl/workers/rollout/llm_server.py:409-418`), alignment from prompt start (`XV/verl/experimental/agent_loop/agent_loop.py:801-826`), every layer and position replayed. Crucible's `TorchEngine` carries `resume_routing` correctly (`C/train/engine.py:352-356, 383-384`; `runtime.py:270`); the SGLang adapter does not (H5, also [ran `t2_sglang_r3_rows`]: 12 rows for 3 tokens, 27 for 6 after resume; row 0 = prompt position 0 is attributed to completion token 0). Replay scope: §1.7.

### 2.4 Masked (infra-failed) rows

| | MiMo | Crucible |
|---|---|---|
| Marking | infra category → reward −999 (`XV/recipes/general/agent_loop.py:76-87`), `extra.is_infra` | `reward=None` (`runtime.py:356-360`) |
| Group baseline | fill with valid group mean, then A = 0 (`XV/core_algos.py:304-350`); optional singleton uid (`DROP_INFRA_FROM_GROUP=1`, `trainer_base.py:2023-2033`) | excluded from every statistic (`objectives.py:90-110`) |
| Loss denominator | **response_mask not cleared**: `batch_num_tokens` = all-reduced `loss_mask.sum()` incl. infra rows (`XV/verl/workers/engine/fsdp/transformer_impl.py:677-681`); prompt-mean weights count infra tokens (`XV/core_algos.py:1160-1182`) | masked rows yield no items (`trainer.py:458-459`), so `g_tokens` excludes them (`:603-605`) |
| Zero-variance filter | `_classify_group` ignores infra rows; ≤ 1 valid member = no signal, evicted (`replay_buffer.py:257-269`) | on tag rewards, `len(rs) > 1 and len(set(rs)) == 1` (`scheduler.py:172-178`) |
| Failed session | evicts and refills the whole group (`replay_buffer.py:643-651`) | `group_backfill` pad (default) / refill_member / drop_group (`scheduler.py:130-150`) |

[ran `t2_infra_dilution`] MiMo's `compute_grpo_outcome_advantage` and `compute_prompt_loss_weights` from source, 2 prompts × 4 rows × 10 tokens, one infra row in prompt A. Advantages identical ([.333, −.667, .333, 0 | −.5, −.5, .5, .5]). Per-token weight of prompt A's valid rows: MiMo 0.0125, MiMo drop-infra 0.0111, **Crucible 0.0167**; token-mean denominator MiMo 80, Crucible 70. MiMo shrinks the gradient of every prompt holding an infra row by 25 % (by 1/3 batch-wide in drop-infra mode, where the infra row becomes a phantom prompt). **Crucible is right here.**

[ran `t2_masked_tags`] Two Crucible gaps (D-T2-5, P2): a record returned with `mask_sample=True` or `reward=None` by a custom `rollout_fn` is finished as `finished` (`runtime.py:361-364`), so the zero-variance filter, backfill, `masked_rollouts` and the circuit breaker miss it: [1, 1, 1, masked] is kept; one valid member + three pool-masked under `pad` is kept (MiMo evicts both); an all-masked group is evicted but labelled `zero_variance`. Builtin rollout fns return `None` instead, so only custom fns hit it.

### 2.5 Group grading and fallback

`XV` has no GAR or GRS. Its only groupwise grader is the webdev group pick (`XV/recipes/design/webdev/group_reward.py:107-268`, called from `trainer_base.py:1888-1915`): synchronous at advantage time, 1500 s budget; a late or failed slice marks its rows INVALID (valid-mean fill → A = 0); `filter_groups` is off for that arm (`webdev.yaml:82-83`). Crucible's stage starts at terminal time with `group_deadline_s` (30 s), cancels late results, falls back to raw advantages with `group_credit_fallback` (`runtime.py:365-416`), and replaces the statistic only when every scored member has a stage advantage (`trainer.py:409-416`).

| Difference | Severity |
|---|---|
| H1: the documented `zero_hacks → grs → gar` chain zeroes the group | P0 |
| `filter_zero_variance` reads raw tag rewards *before* the stage, so it evicts all-pass groups GRS / BT / tournament would separate ([ran `t2_zero_var_vs_group_stage`]: GRS would give [1.0, 0.54, 0.30, 0.45]); no guard (D-T2-4) | P2 |
| Fallback trains a group on raw rewards in the same batch as groups on stage-scale rewards; MiMo's fallback is zero gradient per affected row, Crucible fails the whole group on one bad judge value (`group_ops.py:75, 100, 127, 149`) (D-T2-7) | P2 |


## 3. The verl bridge against local verl `5eb4826` (task 3)

### 3.1 Registration and dispatch: works [ran `t3_bridge_probe` A/B]

| Check | verl | Crucible | Result |
|---|---|---|---|
| `register_adv_est(name)(fn)` | raises only when the name holds a *different* fn (`V/core_algos.py:116-134`) | `register_verl()` registers module-level `crucible_delta` (`verl_bridge.py:57-60`) | registered; idempotent |
| Name | `AlgoConfig.adv_estimator: str`, no allow-list (`V/verl/trainer/config/algorithm.py:734`) | `algorithm.adv_estimator=crucible_delta` | accepted |
| Call | kwargs `token_level_rewards`, `response_mask`, `config`, `index=uid` (np object array), `reward_baselines` if present (`V/verl/trainer/ppo/ray_trainer.py:249-279`; v1: `v1/utils.py:167-175`) | `crucible_delta(token_level_rewards, response_mask, index=None, config=None, **_)` (`verl_bridge.py:36`) | compatible; `set(list(uids))` fine |
| Return | `(advantages, returns)` [B, T] | `(adv, adv)` | as verl grpo |
| vs `grpo(norm_adv_by_std_in_grpo=False)` | `V/core_algos.py:268-333` | no turn credit | 16 rows, groups of 4: **max |Δ| 0.0** |

### 3.2 Behavioural differences [ran `t3_bridge_probe`]

| Case | verl grpo | `crucible_delta` | Note |
|---|---|---|---|
| Group of 1 | A = R | A = 0 | document |
| Masked row (all-zero mask, R 0) in [1, 1, 0, masked] | counted: A₀ = 0.5 | excluded: A₀ = 0.333 | Crucible's contract. `XV`'s −999 sentinel is not recognised and would enter the baseline |
| Response ends on mask-0 tokens | sums the row: R kept | reads R at the last sampled token: **A = 0** | H6 |
| Turn-credit encoding (`token_rewards`) | n/a | [0.3, 0.3, 0, 0, −0.1, 1.0] → A [0.8, 0.8, 0, 0, 0.4, 0.5] | last call's credit dropped on its final token, by design (`verl_bridge.py:9`) |
| `use_kl_in_reward` (scores − β·kl per token) | sum | −β·kl_t read as turn credit; the last token's kl overwritten by R | Δ 0.0171 at β 0.1, silent (P2) |
| Two compaction segments = two rows, one uid | n/a | the rollout counts twice in the baseline (0.5 vs 0.333) | a custom estimator gets no `non_tensor_batch`, so it cannot group by `crucible_rollout` (P2) |

### 3.3 Would it run? No

1. `verl_agent_loop_output()` (`adapters.py:22-26`) → `AgentLoopOutput(**d)`: `ValidationError: metrics field required` (`V/.../agent_loop.py:88-115`, `metrics: AgentLoopMetrics` has no default). No `routed_experts`, `min/max_global_steps`. No Crucible class subclasses `AgentLoopBase` or uses `@register` (`agent_loop.py:227, 487`); M12 §8 lists `CrucibleAgentLoop` as remaining.
2. Turn credit has no channel. An agent loop delivers a scalar `reward_score` placed at one position as `rm_scores` (`agent_loop.py:143-148, 1110-1116`); `extra_fields` land in `non_tensor_batch` (`:1138-1148`) but `compute_advantage` passes only `token_level_rewards`, `response_mask`, `config`, `uid`, `reward_baselines`. `token_rewards()` has no caller outside `T/test_train_bridges.py:49-53`. [ran F] `rm_scores = [0, 0, 0, 0, 0, 1.0]`. `M12-train.md:142` overstates it (H7).
3. `T/test_train_bridges.py:33-61` never imports verl, never builds an `AgentLoopOutput`, never goes through `compute_advantage`: vacuous for the verl contract (P1). The stub loader (`t3_verl_loader.py`) shows a CPU test against verl's real `core_algos.py` is feasible.

## 4. What verl and MiMo-verl do that Crucible lacks (task 4)

### 4.1 Global normalisation: token-mean is right

| | verl | Crucible | [ran `t3_agg_probe`] |
|---|---|---|---|
| token-mean | `batch_num_tokens = all_reduce(SUM, loss_mask.sum())` per mini batch (`V/verl/workers/engine/fsdp/transformer_impl.py:732-738`); loss = Σ / N · dp_size; DDP/FSDP gradient mean | `g_tokens` over all items before partitioning (`trainer.py:603-605`); J = Σ / g_tokens per micro batch (`:640`); gradients all-reduced SUM (`:709-716`), FSDP `set_gradient_divide_factor(1.0)` (`C/train/fsdp.py:97`) | 4 ranks: loss and gradient Δ **0.0**; both equal the full-batch `aggregate` (2.8e-17) |
| Rejected tokens | decoupled RS removes them from N (`rollout_corr_helper.py:1048-1061`); bypass keeps them | kept (`trainer.py:603`) | §1.1 item 3 |
| prompt-mean (MiMo default in all four recipes) | per-row weights 1 / (P · tokens_p) once over the optimizer batch, weighted sum per micro batch (`XV/core_algos.py:1160-1182, 1217-1222`); MiMo requires mini batch = train batch (`XV/recipes/general/run_general.sh:38`) | one micro batch for the whole batch; refused under DP (`trainer.py:606-607, 625, 700-701`) | `token_budget=40`: 8 micro batches for token-mean vs **1** for prompt_mean / seq_mean / const_max_len; MiMo weights over 4 parts reproduce Crucible's full-batch prompt_mean to 2.8e-17 (gradient 1.7e-18); verl seq-mean with `global_batch_size` reproduces seq_mean to 2e-10 (H8) |

### 4.2 Other mechanisms

| Mechanism | verl / MiMo | Crucible | Gap |
|---|---|---|---|
| Dynamic bsz | `use_dynamic_bsz`, `ppo_max_token_len_per_gpu` → `rearrange_micro_batches` (`V/verl/workers/engine/utils.py:92-130`) | `batching.micro_batches` FFD (`batching.py:50-64`), `_equalize` for FSDP layer mode | equivalent in spirit |
| Remove padding | `use_remove_padding: true` (`XV/recipes/general/config/general.yaml:69`) | `HFPolicy.batch_token_logprobs` right-pads to L_max (`C/train/hf.py:122-130`) while the budget counts unpadded tokens | [ran `t4_padding_budget`] [3000] + 10 × [100], budget 4096 → one micro batch, padded tensor **33,000** tokens (8.1×). P2 (OOM on long-tail agentic batches) |
| DP balancing | Karmarkar–Karp on workload 24576·s + s² (`V/verl/utils/seqlen_balancing.py:27-46, 49, 213`), group-balanced partitions (`ray_trainer.py:1182-1225`) | LPT greedy on linear tokens (`batching.py:26-36`; the docstring says "Karmarkar–Karp-style") | [ran `t3_balance_probe`] long tail dp = 4: max/mean workload 1.461 vs **1.350**. P2 |
| Mini batches / epochs | `ppo_mini_batch_size` slices, `ppo_epochs`, shuffle, seed (`ray_trainer.py:1351-1366`) | `updates_per_batch` repeats the **whole** batch (`trainer.py:621-625`) | [ran] 8 items × 4 updates = 32 forwards, 4 steps: epochs, not slices. `M12-train.md:53` says slices. P2 |
| Entropy | exact, from logits, chunked (`entropy_from_logits_with_chunking`, `XV/recipes/code/config/train.yaml:99-100`); `entropy_coeff` term | −mean behaviour logprob drives the monitor and `entropy_adaptive` (`runtime.py:629`, `trainer.py:368-371`); biased under top-p; no entropy term | P2 |
| KL | k1/k2/k3/low_var_kl (clamped), `+` straight-through, in loss or in reward with an adaptive controller (`V/core_algos.py:153, 2188-2251`) | k3 in the loss only, unclamped (`trainer.py:534-535`) | P2 (§1.2; MiMo turns KL off, `general.yaml:55, 86`) |
| Old logprob source | decoupled vs `bypass_mode` (`rollout_corr_helper.py:1109-1144`) | `old_logprob_source` None / recompute_current / rollout (`mismatch.py:92-98`) | equivalent |
| Masked rows | §2.4 | §2.4 | Crucible right |

## 5. All defects

| ID | Sev | Defect | Crucible | Reference | Evidence |
|---|---|---|---|---|---|
| H1 | **P0** | `zero_hacks → grs → gar` zeroes mixed groups | `train/group_ops.py:5-6, 92, 112-113`; `trainer.py:408-416` | `credit/gar.py:51` | [ran `t1_gar`] |
| H2 | **P0** | `QDQ(model=)` freezes QDQ'd master weights | `train/weights.py:150-158` | article Fig. 6 | [ran `t1_qdq`] 49/28,672 moved |
| H3 | **P0** | vLLM nucleus rebuilt from processed (renormalized) logprobs is cut short; trainer π_θ ≠ π_rollout | `train/vllm_http.py:136-158`; fixture `T/test_vllm_engine.py:132-142` | OpenEnv RFC 006:241 | [ran `t1_nucleus`] 3340/4000 silent, 0.13 nats |
| H4 | P1 | Row-level terms switch to per-call rows under top_p < 1 or R3 | `trainer.py:270-274, 460-481` | `XV/recipes/general/token_trace.py:73-97` | [ran `t1_item_unit`] grad-norm +36 % |
| H5 | P1 | SGLang R3 rows misread (prompt rows as completion routes; layers flattened; re-appended on resume); vacuous fixture | `train/sglang_http.py:80-83`; `engine.py:286`; `T/test_sglang_engine.py:26` | `SL/sglang_rollout.py:191-238`; `XV/llm_server.py:409-418` | [ran `t1_r3_sglang`, `t2_sglang_r3_rows`] |
| H6 | P1 | `crucible_delta` loses R when a row ends on observation tokens | `bridges/verl_bridge.py:41-44` | verl grpo row sum | [ran E] |
| H7 | P1 | verl bridge cannot run; turn credit has no channel; doc overstates | `bridges/adapters.py:22-26`; `verl_bridge.py:23-33`; `M12-train.md:142` | `V/agent_loop.py:88-115, 143-148`; `ray_trainer.py:249-279` | [ran F] |
| H8 | P1 | prompt-mean / seq-mean / const_max_len: one micro batch, no DP | `trainer.py:606-607, 625, 700-701` | `XV/core_algos.py:1160-1182` | [ran agg 1, 3, 4] |
| H9 | P1 | `capture_incomplete` never masked; missing logprobs train as 0.0 | `world_env.py:189`; `interop/nemo_gym/client.py:155`; `bridges/common.py:79` | `XV` TokenTrace | [ran `t2_incomplete_capture`] |
| D-T3d | P1 | verl bridge test is vacuous | `T/test_train_bridges.py:33-61` | — | read |
| D1 | P2 | PPO-family dual clip off by default; no log-ratio clamp | `mismatch.py:82-86, 112-113` | `V/core_algos.py:1323, 1338` | [ran] 1.47e9 at log r = 25 |
| D2 | P2 | `const_max_len` is the batch maximum, not a constant; all rows counted | `objectives.py:59-61`; `trainer.py:638` | `V/core_algos.py:1183`; `actor.py:164` | [ran] 7/8 |
| D3 | P2 | k3 KL unclamped, no `k3+`, no kl-in-reward | `trainer.py:534-535` | `V/core_algos.py:2188-2251` | [ran] 7.2e10 |
| D4 | P2 | GAE / critic: no whitening option, value loss without clip | `objectives.py:148-207`; `trainer.py:249-268` | `V/core_algos.py:216-262, 2125-2186` | [ran] raw GAE Δ 5e-8 |
| D5 | P2 | GRPO std is population; reinforce_pp whitens per rollout; n = 1 → 0 | `objectives.py:86, 105-107` | verl grpo / reinforce_plus_plus | [ran I] 0.219, 0.093 |
| D6 | P2 | Two GAR implementations, different rules and defaults; `group_ops.gar` binarizes partial credit; capped path moves failing members | `group_ops.py:83-116`; `credit/gar.py:34-64` | article GAR | [ran `t1_gar`] |
| D7 | P2 | MXFP4 ties round down, not to even (bf16: 0.47 % of elements); default QDQ selects every matrix | `weights.py:121, 131` | OCP MX v1.0 | [ran `t1_qdq`] |
| D8 | P2 | R3 replays completion positions only; `HFPolicy` / FSDP have no R3; no continuation row check | `engine.py:300-305`; `hf.py:119`; `fsdp.py:74` | `SL/cp_utils.py:472`; `SL/sglang_rollout.py:193-197` | read |
| H10 | P2 | τ = 4 is one version looser than MiMo's T = 4 | `scheduler.py:47` | `XV/replay_buffer.py:632-667` | [ran `t2_mimo_rb`] |
| D-T2-4 | P2 | zero-variance filter evicts all-pass groups before the group stage | `scheduler.py:172-178` | `XV/webdev.yaml:82-83` | [ran] |
| D-T2-5 | P2 | custom-`rollout_fn` masked records counted as finished/valid | `runtime.py:361-364` | `XV/replay_buffer.py:257-269` | [ran `t2_masked_tags`] |
| D-T2-7 | P2 | group-stage fallback mixes raw and stage-scale rewards; one bad judge value fails the group | `runtime.py:384-416`; `group_ops.py:75, 100, 127, 149` | `XV/group_reward.py:225-247` | read |
| D-T2-8 | P2 note | call-level re-issue default; MiMo always token-level | `runtime.py:68` | `XV/llm_server.py:393-441` | read |
| D-T3e | P2 | `crucible_delta` under kl-in-reward and multi-segment rows | `verl_bridge.py:47-53` | `ray_trainer.py` `apply_kl_penalty` | [ran G, H] |
| D-T4b | P2 | token budget ignores padding | `batching.py:50-64`; `hf.py:122-130` | `use_remove_padding` | [ran] 8.1× |
| D-T4c | P2 | `updates_per_batch` = epochs, doc says slices | `trainer.py:621-625`; `M12-train.md:53` | `ray_trainer.py:1351-1366` | [ran] |
| D-T4f | P2 | DP balance on linear tokens, LPT | `batching.py:26-36` | `seqlen_balancing.py` | [ran] 1.461 vs 1.350 |
| D-T4g | P2 | entropy = behaviour-logprob estimate | `trainer.py:368-371`; `runtime.py:629` | chunked entropy | read |

Where Crucible is right and MiMo is not: infra rows dilute MiMo's token and prompt denominators by 25–33 % (§2.4); MiMo leaves partial-rollout drift uncorrected by default; MiMo's checkpoint resume resets the staleness clock; GAE carries observation-position rewards instead of dropping them.

## 6. Picks

| # | Pick | Source | Verdict | Effort |
|---|---|---|---|---|
| 1 | One GAR (`credit/gar.py`); `group_ops.gar` computes on received rewards with a GRS-aware passing predicate; a test of the documented three-stage chain | article; `credit/gar.py` | PATTERN | S |
| 2 | QDQ: never overwrite master weights; publish the QDQ'd copy and fake-quantize the trainer forward (STE) when bit parity is wanted; RNE ties; default `include=("*experts*",)` | Kimi K3 QAT / OCP MX v1.0 | PATTERN | M |
| 3 | `_nuclei`: nucleus = finite entries of processed logprobs (require a −inf sentinel in top-k); refuse top-p in raw mode; fixture with renormalized logprobs | vLLM processed mode | PATTERN | S |
| 4 | `seq_id` per item; row-level terms (seq-mean, GSPO, MIS-PO / HisPO gates, const_max_len rows) aggregate over (rollout, actor, segment) | `XV` one row per trajectory | PATTERN | M |
| 5 | Per-row global loss weights for prompt_mean / seq_mean / const_max_len (micro batches + DP); `loss_scale_factor` knob | `XV/core_algos.py:1160-1222`; `V agg_loss` | PORT | S |
| 6 | SGLang R3: slime's row contract (`routed_experts_start_len`, rows = tokens − 1, per-layer arrays), prompt routes replayed, continuation check; R3 in `HFPolicy` | `SL/sglang_rollout.py:190-238`; `XV/llm_server.py:409-418` | PORT | M |
| 7 | Gate the trainer on `valid_for_training()`; build M12 records with `mode="train"` | M10 `record.build(mode="train")` | PATTERN | S |
| 8 | `CrucibleAgentLoop(AgentLoopBase)` with `metrics`, `min/max_global_steps`, `routed_experts`; a reward hook writing `token_rewards()` into `rm_scores`; `crucible_delta` takes R as the row sum after the last sampled token and knows `invalid_reward_value`; a test through verl's real `compute_advantage` | `V/agent_loop.py`, `ray_trainer.py` | DEPEND | M |
| 9 | Dual clip c = 3 default and ±20 log-ratio clamp for the PPO family; GSPO log-s clamp 10; k3 clamps and `k3+` | `V/core_algos.py:1323-1338, 1590, 2188-2251` | PORT | S |
| 10 | Workload Karmarkar–Karp DP balancing; padding-aware budget or varlen packing; mini-batch slicing distinct from epochs | `V/seqlen_balancing.py`; `ray_trainer.py:1351-1366` | PORT | M |
| 11 | Exact chunked entropy for the monitor and `entropy_adaptive` | `V entropy_from_logits_with_chunking` | PATTERN | S |
| 12 | Map τ = T − 1 for MiMo configs (or count span); zero-variance filter after the group stage; mark custom masked records `masked` | `XV/replay_buffer.py` | PATTERN | S |
| 13 | Bessel-std option, token-weighted reinforce_pp whitening, `reinforce_plus_plus_baseline`, optional GAE whitening, clipped value loss | `V/core_algos.py` | PORT | S |

## 7. Probe index

| Probe | What | Key output |
|---|---|---|
| `t1_loader.py` | loads `V` / `XV` `core_algos.py` + `rollout_corr_helper.py` with stubs | both registries load |
| `t1_corrections.py` | 12 correction pairs × 4 agg modes × normal/extreme | §1.1 table |
| `t1_kl_gae.py` | k3 vs low_var_kl / k3+; GAE vs verl | 0 / 7.2e10; 5e-8; 0.5 on obs rewards |
| `t1_gar.py` | GAR/GRS chains | [0, 0, 0, 0] |
| `t1_qdq.py` | OCP RNE reference; master-weight freeze | 0.47 % bf16; 49/28,672 |
| `t1_nucleus.py`, `t1_nucleus_k.py` | nucleus vs HF; `_nuclei` on processed logprobs | 0/4000; 2414–3340/4000 |
| `t1_r3_sglang.py` | SGLang row contract | 13 rows / 4 tokens |
| `t1_item_unit.py` | top_p switches items | 2 → 4 items, grad-norm 0.102 → 0.139 |
| `t2_staleness.py`, `t2_mimo_rb.py` | MiMo ReplayBufferAsync vs Crucible sim | lag 3 vs 4 |
| `t2_infra_dilution.py` | MiMo infra rows in denominators | 0.0125 vs 0.0167 |
| `t2_masked_tags.py`, `t2_zero_var_vs_group_stage.py`, `t2_incomplete_capture.py`, `t2_sglang_r3_rows.py` | masked tags, filter order, missing logprobs, R3 rows | §2 |
| `t3_bridge_probe.py`, `t3_agg_probe.py`, `t3_balance_probe.py`, `t3_verl_loader.py`, `t4_padding_budget.out` | bridge, aggregation split, DP balance, padding | §3, §4 |
