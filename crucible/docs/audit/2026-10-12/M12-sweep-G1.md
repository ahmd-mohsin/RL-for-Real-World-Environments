# M12 `train` sweep, group 1: the articles as a requirement ledger, with golden probes (2026-10-13)

Read-only. Crucible HEAD `3b522cb`. One reader. `CR` = `crucible/src/crucible`, `T` = `crucible/tests/unit`.

**Sources.**
- **Q**: the qingkeai article, `RL-Environments/07-articles/qingkeai-2026-agentic-rl-infra.md`. This is the Chinese original and the authoritative text. `Q:L123` is a line of that file.
- **Formulas.** The md lost every display formula, so all 56 were re-extracted from `raw/qingkeai-2026-agentic-rl-infra.html` (`<div class="language-math">`, in document order). `F1`…`F56` below are those formulas. The extract is `scratchpad/m12/g1/formulas.md`.
- **Z**: the Zhihu digest.
- **MH**: `hf-space-fineenvs-multi-harness-rl.md`.
- Every other article in `07-articles/` was scanned for training-loop content: hf-blog, hf-space guide, multi-harness stub, pi-verifiers-v1, pi swarm, PM / Karotte ×3, git-swarm, repo2rlenv, prime-agent ×4. Only MH, pi-verifiers (branches as samples) and Continual Harness (`prime-agent/2605.09998.txt:2067-2071`: offline GRPO grouped per state, β = 0.04) bear on training. The rest are environment-side.

**Probes.**
- Location: `scratchpad/m12/g1/probes_{math,sched,credit,misc,extra}.py` (scratchpad = this session's `/private/tmp/.../scratchpad`).
- Run: from `crucible/`, `PYTHONPATH=src .venv/bin/python <file>`.
- Each probe calls Crucible's real code and compares it to the article's formula or numbers.
- Every probe cited below ran. A [ran] tag marks a result taken from a probe; [code] marks a reading of the code only.
- Totals:

| File | Pass | Note |
|---|---|---|
| math | 51/59 | 2 of the 8 FAILs are float artefacts at band edges: `exp(log 0.5) = 0.49999999999999994`. Re-probed with exact log construction, both pass. |
| sched | 14/15 | |
| credit | 19/33 | |
| misc | 5/6 | |
| extra | all printed | |

Severity: **P0** = wrong gradient or reward, or silent training on bad data. **P1** = contract break or vacuous test. **P2** = gap.

---

## Headline defects

| # | Sev | Defect | Evidence |
|---|---|---|---|
| H1 | **P0** | **Truncation bootstrap adds V(s_T) to the reward instead of replacing it** (critic mode). The article says R̂ = V_φ(s_T) for a truncated rollout (F52, Q:L636-638); AReaL zeroes the task reward and bootstraps (`06-frameworks-and-env-code/infra-inclusionai-areal/tests/test_ppo_actor_truncation.py:282-300`). Crucible writes `rewards[-1] = r.reward` (`CR/train/trainer.py:229`), then adds `bootstrap_value = v[-1]` as the terminal value (`trainer.py:236`, `CR/train/objectives.py:200`). Reachable: World limit stops (`max_steps`, `timeout`, …) set `truncated=True` (`CR/actors/policy.py:653,658`) and are still scored ("score", or "penalty:<c>"). | [ran TRUNC-tr1.0] truncated, R = 1, V ≡ 0.5: the critic target at the last token is **1.5**, outside the reward range, and the policy advantage is **+1.0** instead of 0. [ran TRUNC-R-1] R = −1 (a penalty) gives −0.5. The only test uses R = 0 (`T/test_train_critic.py:53-61`; `T/test_train_math.py:140-141`), where add and replace agree. |
| H2 | **P0** (opt-in) | **`max_offpolicy_frac` breaks the staleness bound under the default `wait`, and deadlocks the disaggregated topology.** The cap removes stale groups from the candidates (`CR/train/scheduler.py:205-210`) *before* the at-bound rule (`:223`) can train or evict them. `wait` only blocks on live rollouts (`:167-171`), so they keep aging. | [ran OFFPOL extra] τ = 3, colocated_partial, frac 0.25: trained staleness climbs 0, 1, …, **8** (frac 0.5: 4). disaggregated_stream, frac 0.25 or 0.5: **1 step of 30**, then `need_more` forever. The refill is admitted while training runs on the old version, so the S·B·n cap fills with off-policy groups and no on-policy group can enter. The test's assertion is always true (H6). M12 checklist #25 says "done". |
| H3 | **P1** | **GSPO is GSPO only with `loss_agg="seq_ratio"`, and nothing sets or enforces it.** `TrainerConfig.loss_agg` defaults to `token_mean_global` for every correction (`CR/train/trainer.py:44`). | [ran GSPO-seq_ratio] value and gradients equal a reference GSPO (F36) exactly. [ran GSPO-token_mean_global] value −0.050 vs 0.250, and the long sequence's gradient is 1.4× the reference. [ran GSPO-dflt] `TrainerConfig(correction=gspo).loss_agg == 'token_mean_global'`. HisPO has the same trap (H9). |
| H4 | **P1** | **Critic mode silently drops every reward-shaping term**: speed, latency (`speed_coef`, `latency_coef`), `Turn.step_reward` (process, reward-to-go). `_items` calls `_shaped_rewards` only on the group branch (`trainer.py:395-400`). `_gae_advantages` puts only `r.reward` at the last token (`:228-229`). | [ran SPEED] two passing rollouts with active 1 s vs 10 s, `speed_coef = 0.5`. Group mode: adv +0.409 / −0.409. Critic mode: **0.605 / 0.605**. [ran CRIT-step] a −0.5 step reward leaves the critic-mode advantages unchanged. Z-9 and MiniMax (Q:L700) need per-action rewards. |
| H5 | **P1** | **No check that the trainer's temperature matches the sampling temperature.** The trainer evaluates π_θ at `TrainerConfig.temperature` (`trainer.py:52, 488-515`). `Turn.sampling["temperature"]` is never read. MH:L806 states the two must match. Otherwise r_dis is biased on every token, and DIS masks or reweights it silently. | [ran TEMP-guard] records sampled at T = 0.7 train at T = 1.0 with no error. |
| H6 | **P1** | **Three "golden" tests are vacuous or blind.** | (a) `T/test_train_sched.py:82` asserts windowed-FIFO lag ≤ 10 with `admission_staleness=1`, B = 8, n = 1, so max_inflight = B. [ran WFIFO-test] actual max staleness is **0**. (b) `T/test_train_sched.py:106` asserts `≤ max(1, len(b)//4)*4` = ≤ 16 of 16 rollouts, which is always true [ran OFFPOL]. It hides H2. (c) The truncation tests use R = 0 only, which hides H1. Also: the GSPO golden checks s = 1 only, with no aggregation (`T/test_train_math.py:77-84`). |
| H7 | **P1** | **The online interceptor misses the exact git back-link the article documents** (Qwen: re-add a remote, then fetch; Q:L808). `REMOTE_FETCH` needs a URL right after `clone/fetch/pull` (`CR/world/intercept.py:50-55`). There is no repo-slug-plus-network-keyword rule. | [ran INTERCEPT] `git remote add origin https://github.com/o/r && git fetch origin` and `git remote add up …; git pull up main` both **miss**. `git clone URL` and `curl URL` hit. The egress guard (`CR/world/guard/guard.py:345`) may still block at network level when it is on. |
| H8 | P2 | **Two GAR implementations disagree, and the wired one is not the article's.** `CR/train/group_ops.py:83-116` matches F55 exactly (λ = Σ_P A / Σ_P fA, cap, re-centring). `CR/credit/gar.py:34-64` is an interpolation A + lam·(M·f/Σf − A) with no amplification cap, and `gar_group_fn` defaults to lam = 0.5 (`:90`). The documented hack chain uses that one (`CR/credit/hack_detector.py:13`). | [ran GAR-ops, GAR-cap] pass. [ran GAR-cr0.5] gives [0.339, 0.232, 0.179] vs the article's [0.429, 0.214, 0.107]. [ran GAR-crcap] at lam = 1 the top passer is amplified 2.5×, where the article's cap at 2 gives 2.15×. |
| H9 | P2 | **HisPO's G·max\|y\| denominator is per batch, not per group.** `objectives.py:59-61` divides by rows × the batch-global max. It is also not the default for `hispo`. | [ran HISPO-den] two groups with max \|y\| 4 and 10: 0.550 vs the article's 0.775. |
| H10 | P2 | **Kimi verbosity rule is not the article's, and the M12 doc says it is.** Article: a candidate longer than σ·ℓ₀ auto-loses the comparison, with ℓ₀ from the cold-start model (Q:L744). Code: a bounded penalty ≤ 0.2, anchor = the group's 0.3-quantile of passing lengths, applied only when pass rate > 0.5 (`CR/credit/shaping.py:68-103`; a deliberate M6 change). `docs/modules/M12-train.md:35` still says "longer than σ·ℓ₀ loses". | [ran KIMI-verb] 300 tokens against a budget of 150 → reward **0.80**. [ran KIMI-verb2] at pass rate 0.5 → 1.0, untouched. [ran KIMI-tour] the tournament has no length rule. |
| H11 | P2 | **DAPO overlong shaping is a function nobody calls.** `objectives.overlong_penalty` (`:136-141`) is golden-correct [ran DAPO-soft], but a grep of `CR` finds no caller. There is no overlong-filter switch for token-env length truncation; `runtime.py:627` only counts it. `M12-train.md` §5 says "DAPO overlong shaping is available". | grep; [ran DAPO-soft] 16,384 → 0, 18,432 → −0.5, 20,480 → −1, continuous. |
| H12 | P2 | **Smaller divergences.** | (a) Ring σ alone does not bound staleness: buffered groups age without being cut [ran RING-s0: σ = 0 trains staleness 1]. `_cut_live` also counts `grading` rollouts as cut (`runtime.py:802-803`). (b) The admission cap S·B·n holds, but the docstring's "trained within about S steps" (`scheduler.py:15-16`) does not [ran LING: S = 2 → staleness 5, S = 4 → 10 with `staleness_strategy="none"`]. (c) Sequence-level gates (MIS-PO ρ̄, HisPO seq gate, GSPO s_i) are computed per training row (`mismatch.py:143-152,171`). They become per-call when items fall back to calls, which happens for top-p < 1 or R3 (`trainer.py:270-276,460`), exactly the MoE settings where these corrections are used [code]. (d) `gae` carries a reward on an observation position to the *next* action (`objectives.py:169-176`), but F49 puts it in the δ of the action that earned it [ran SAO-stepr: off by −r(1−γλ)]. Latent today, because of H4. (e) The CISPO unified mask M_it (F46) and CISPO+MIS composition (M2) are missing [ran CISPO-mask]. (f) MIS-PO with `old_logprob_source="rollout"` gives x ≡ 1 and filters nothing, without a warning [ran MISPO-src]. |

**Confirmed correct**, golden against the article's numbers [ran]:
- KPop binary KL: 4.0473e-5 and 0.4534; the approximation gives 4.047e-5 and 0.174; the mask keeps ratio 5 and masks ratio 2 at φ = 1e-3 to 0.1.
- IcePop: M(k) = k calibration, so the gradient is r_dis·r_stale·Â in band and 0 outside [0.5, 5].
- TIS: one-sided.
- DIS: open band, sign-blind, gradient r·Â, both SAO bands.
- MiMo four bounds: inclusive, selected by the sign of Â; entropy adaptation moves the right way.
- MIS-PO:
  - token band [0.5, 2] and trajectory band [0.996, 1.001], asymmetric in log space;
  - e^{−0.004T} = 0.670 / 0.0183 / 2.06e-9;
  - a T = 5000 trajectory sitting at the edge passes;
  - weight 1, and r_stale is ignored.
- HisPO: gates use r_dis only; three bounds; dual clip at (1 + ε_neg_high)·Â.
- CISPO:
  - in-band gradient equals PPO's (autograd);
  - out of band the coefficient is held at 1.28 while PPO's gradient is 0;
  - without sg the gradient is r̂(1 + log π) = −6.39 at p = 1e-3;
  - M1 vs M2 bounds.
- GSPO under `seq_ratio`; sequence-level clipping.
- PPO clip-higher and dual clip c = 10: A < 0, r = 20 gives objective −10 and gradient 0; A > 0 is untouched.
- DAPO token vs sample weights: 5e-3 / 5e-5 vs 9.9e-5.
- SAO skip-observation GAE equals F49 directly.
- CompactionRL (γλ)^{N>s}.
- λ_policy = 1 − 1/(1.5·l).
- γ = λ = 1 gives Â = R − V.
- TTUR K = 2.
- GLM τ = 3, w′ = 10 drop.
- Ling cap.
- Windowed FIFO: lag 30 under greedy, 1 at W = 0.3N; trainer idle 16 s → 462 s.
- GRS.
- PARL critical steps and DAG path.
- Qwen format penalty on the offending call only.
- MiniMax reward-to-go.
- MXFP4 QDQ.
- Top-p renormalization.
- k = 8 rerun screen.
- Noise curriculum.
- **FP32 LM head = model logits exactly** (Qwen3 and Llama, transformers 5.19). The final hidden state is post-norm, so the FP32 path is sound.

---

## 1. Corrections to the prior studies

| Prior claim | Finding |
|---|---|
| `docs/research/m11-m12/06-article-systems.md:262`: "the article's worked example: batch = 8, W = 4 → lag bound 8 + 3 − 1 = 10" | **Not in the Chinese text.** Q:L190 says only "Max Off-Policy Lag is bounded by batch and window together". grep finds no such numbers in the md or the raw HTML. It probably came from the English translation's interactive widget, which is not saved, so its provenance cannot be checked. The test that asserts it is vacuous (H6a). |
| `06:368`: MiMo prompt-mean = "token-average inside each response, then across prompts" (copied from Q:L268) | MiMo's own verl fork defines prompt-mean as the **token mean over the whole prompt group**, then the mean over prompts (`06-frameworks-and-env-code/xiaomimimo-verl/tests/trainer/ppo/test_prompt_mean_loss_on_cpu.py:39-47,63-66`). Crucible's `prompt_mean` (`objectives.py:48-58`) matches the fork [ran MIMO-pm: 0.60, where the article's wording would give 0.75]. **No defect.** The article's paraphrase is the loose one. |
| `docs/research/m13-audit/01-article-beyond-52.md` A2-19: acceptance "length σ·ℓ₀ + 1 loses every comparison" | Not met. The bounded penalty replaced it (H10). The M12 doc row still claims the article's rule. |
| `M12-train.md` §9 #25 `max_offpolicy_frac` "done" | Breaks τ and deadlocks (H2); the test is vacuous. |
| #28 "windowed lag ≤ 10" "done" | The test is vacuous (H6a). The mechanism itself works [ran WFIFO-sweep]. |
| #48 "truncation bootstrap" "done" | Wrong for R ≠ 0 (H1). |
| §5: "each correction … with the article's numbers as golden tests" | KPop, DIS, MIS-PO, CISPO and DAPO are article goldens. HisPO's −1.28 is Crucible's own reading: the article gives no number. GSPO's test is at s = 1 only. |
| §5: "DAPO overlong shaping is available" | Unwired (H11). |
| `docs/research/2026-10-09-agentic-rl-infra-survey.md:102` (Kimi: candidates longer than σ·ℓ₀ lose automatically) | Accurate to the article. The code diverged later (H10). |

---

## 2. Requirement ledger

Status: **done** / **partial** / **missing** / **WRONG**.

Rows marked "(doc)" take their status from `M12-train.md` and were not re-probed by G1. They are systems items owned by other groups.

### §1–§2 framing (Q:L25-90)

| id | Art. | Claim | Crucible | Status | Evidence |
|---|---|---|---|---|---|
| P-1 | Q:L31-36 | Four planes: control, data, state, weight | M12 doc §1; `CR/train/state.py` | partial (doc) | checklist #1 |
| P-2 | Q:L21, L50 | Long tail: about 5% of samples take about 80% of generation; MiMo: 90× token spread, 66× active-time spread across 25 sources | `sim.default_shapes` (`sim.py:41-51`); per-source len/active monitors | done (doc) | — |
| P-3 | Q:L68-76, F3-F4 | Token ratio; the sequence ratio ∏r_t explodes with length | GSPO s_i, MIS-PO ρ̄ | done | see ST-13, MM-9 |
| P-4 | Q:L62, L346 | MoE: about 10% of experts flip per update | R3, routing confidence | done (doc) | #36 |

### §3.1 Topology (Q:L94-146)

| id | Art. | Claim | Crucible | Status | Evidence |
|---|---|---|---|---|---|
| T-1 | Q:L44, L100 | Colocated time-sharing; KV dropped for training | `colocated_partial`: abort + pause (`runtime.py:705-708`) | done (doc) | — |
| T-2 | Q:L116-118 | Kimi: weights and optimizer to NVMe/CPU. Hy3: params, optimizer and grads to CPU | `TorchTrainer.offload(cpu\|nvme)` (`trainer.py:136`) | done (doc) | #7 |
| T-3 | Q:L116 | Forward-only reference models on CPU, borrowing the FP32 grad buffer | TeacherPool; teacher offload is A2-13 | partial | — |
| T-4 | Q:L120, L140 | Environments run off the GPU pool, on elastic CPU | `EnvWorkerPool` | done (doc) | #8 |
| T-5 | Q:L108, L128-134 | Disaggregated pools with a producer–consumer data pool | `disaggregated_stream` | done | [ran LING, WFIFO] |
| T-6 | Q:L110 | LongCat DORA: elastic co-location | topology `elastic` (`runtime.py:709-711`) | done (doc) | — |
| T-7 | Q:L128, L182 | GLM: push weights every K steps; reset the optimizer after each update | `publish_every`; `reset_optimizer_on_publish` (`weightsync.py:308-330`) | done (doc) | #13 |
| T-8 | Q:L146 | Fast weight sync: AState P2P, 1T parameters in 10 s; critique of NCCL group rebuilds | bucketed broadcast (§6a) | partial (doc) | RDMA/P2P open (M12 doc §8) |
| T-9 | Q:L146, L480 | TITO gateway | `CR/train/chat.py` exact TITO | done (doc) | — |
| T-10 | Q:L128 | Per-task rollout and reward microservices; 1k+ concurrent rollouts | per-source `rollout_fn`/`reward_fn` maps | done (doc) | C-6 |

### §3.2 Async, partial rollout, long tail (Q:L148-203)

| id | Art. | Claim | Crucible | Status | Evidence |
|---|---|---|---|---|---|
| A-1 | Q:L168 | Kimi: N prompts × K; pause at λNK finished; paused rollouts **resume first** next round | `partial_barrier=completion_frac` (`runtime.py:792-795`) | partial | [ran KIMI-λ] runs. No priority for cut rollouts: grep finds no queue priority; also Z-12 (AReaL re-queues at the head) |
| A-2 | Q:L170 | MiMo: GRPO + async partial; staleness 4; re-prefill cost | `max_staleness=4` (`scheduler.py:47`); `rollout/reprefill_tokens` | done | — |
| A-3 | Q:L174 | DeepSeek: per-task in-flight cap; schedule a prompt once n slots are free; batch-level and prompt-level dispatch rejected | `admit` (`scheduler.py:106-127`); dispatch unit = rollout | done | `T/test_train_sched.py:52-60` |
| A-4 | Q:L176 | DeepSeek: pause at any token boundary; resume across checkpoints without re-prefill | `resume_keeps_kv`, `kv_resumes` | partial (doc) | engine-declared |
| A-5 | Q:L178 | Ring C3PO++: round boundary = token budget Φ | `partial_barrier=token_budget` | done | [ran C3PO-Φ] |
| A-6 | Q:L182 | GLM: stream; batch at a threshold; push every K; reset optimizer | see T-7 | done | — |
| A-7 | Q:L184 | DORA: no batch barrier; per-sample remote workers | `workers.py`; rollout dispatch | done (doc) | — |
| A-8 | Q:L190, Q:L162 | Windowed FIFO [T_i, T_{i+W−1}]: out of order inside the window, blocked across it; W = 0.3N; lag bounded by batch and W | `scheduler.py:195-199`; `window()` `:72-73` | done | [ran WFIFO-sweep] S = 8, B = 8: max lag 30 (greedy), 1 (W = 0.3N), 0 (W = 0.1N), 27 (oldest_first); idle 16 s → 462 s |
| A-9 | (not in Q) | "batch 8, W 4 → lag ≤ 10" | test `T/test_train_sched.py:82` | **vacuous test** | [ran WFIFO-test] max staleness 0 |
| A-10 | Q:L203 | Length bias: per-dataset concurrency; drop early short samples | `inflight_cap`/`mix`; `drop_early_short` (`scheduler.py:179-187`); KS monitor (`monitors.py:73-76`) | done | #29 partial (doc) |
| A-11 | Q:L198 | Ling: partial rollout gives deterministic round boundaries for checkpointing | `state.py` barrier checkpoints | done (doc) | — |
| A-12 | Q:L200 | Staleness bounded by cuts (Ring σ, MiMo 4); only within-round samples are strictly on-policy | `max_cuts` (`runtime.py:798-812`) | partial | [ran RING-s0] σ = 0 → staleness 1 (H12a) |

### §3.3 Staleness (Q:L205-332)

| id | Art. | Claim | Crucible | Status | Evidence |
|---|---|---|---|---|---|
| ST-1 | Q:L225-227, F5 | GLM: record (w0…wk); drop iff w′ − w0 > τ. Example τ = 3, w′ = 10: (6,7,8) dropped, (7,8,9) kept | `Tag.versions`/`staleness` (`pool.py:40-55`); drop (`scheduler.py:159-166`) | done | [ran GLM-drop] (6,7,8) evicted, (7,8,9) trained at staleness 3. Crucible drops the whole group (oldest member, `pool.py:289-297`), which is stricter |
| ST-2 | Q:L229 | Ring: ρ(o) += 1 per unfinished round; purge ρ > σ before the next round | `_cut_live` | partial | counts `grading` rollouts; evicts the group (H12a) |
| ST-3 | Q:L231 | "Cuts = versions behind; σ plays τ" | — | partial | [ran RING-s0/1/2] σ = 0 → 1, σ = 1 → 1, σ = 2 → 2 |
| ST-4 | Q:L235-237, F6 | Ling: \|P_inflight\| ≤ S×B; staleness ≈ ≤ S | `max_inflight = S·B·n` untrained, including the buffer (`scheduler.py:68-70,110`) | done (cap) / WRONG (docstring) | [ran LING-S1/2/4] cap 16/32/64 holds; staleness 1/5/10 (H12b) |
| ST-5 | Q:L239 | Fragments over the bound retire | `wait` / `drop` | done | [ran WAIT-fNone] τ = 2 → 2 |
| ST-6 | Q:L250, F7 | r = r_dis × r_stale | `mismatch.py:3-9,99-101` | done | — |
| ST-7 | Q:L254 | Exact π_θold per token is infeasible under async | DIS default; refusal for multi-version records (`trainer.py:608-613`) | done | — |
| ST-8 | Q:L258-260, F8-F9 | DIS: f(r) = r in (1 − ε_l, 1 + ε_h), else 0; sign-blind | `mismatch.py:129-133` | done | [ran DIS-coding, DIS-math-TIR] |
| ST-9 | Q:L264-268, F10-F12 | MiMo: sg(r); four inclusive bounds chosen by the sign of Â; [0.2, 5] both; ln 0.2 = −1.61, ln 5 = 1.61 | `mismatch.py:134-140` | done | [ran MIMO4-1/2] |
| ST-10 | Q:L268 | Entropy-adaptive: low entropy widens +, tightens −; high reverses | `trainer.py:363-383` | done | [ran MIMO4-3]. The entropy is the behaviour NLL, not the full-vocab entropy |
| ST-11 | Q:L264 | Partial-rollout continuations do not recompute rollout probs | per-call logprobs kept across resumes | done (doc) | #4 |
| ST-12 | Q:L272 | DeepSeek: cap the batch's off-policy share; mask stale tokens in the loss | `max_offpolicy_frac` (`scheduler.py:205-210`); stale-token mask: none | **WRONG** / missing | H2 [ran OFFPOL extra]. The loss has no version-age mask; DIS masks by ratio |
| ST-13 | Q:L276-282, F13-F15 | MIS-PO: x = π_θold/π_rollout; ρ̄ = geometric mean; token [0.5, 2], trajectory [0.996, 1.001]; L = −E[I·I·log π·Â] | `mismatch.py:141-149` | done | [ran MISPO-tok/rej/asym/edge*] |
| ST-14 | Q:L284-288 | e^{−0.004T}: 0.67 / 0.018 / 2e-9 | — | done | [ran MISPO-T*] 0.670 / 0.0183 / 2.06e-9; a T = 5000 trajectory at the edge passes |
| ST-15 | Q:L292 | StepFun stable at about 20 steps of staleness | τ is configurable | n/a | — |
| ST-16 | Q:L298-315, F16-F18 | HisPO: sequence and token gates on r_dis only; full r clipped | `mismatch.py:150-156` | done | [ran HISPO-g*] gates unchanged for r_stale ∈ {0.01, 1, 50} |
| ST-17 | Q:L319-324, F19 | Three bounds; ε_neg_high closes the unbounded A < 0 term | `mismatch.py:157-160` | done | [ran HISPO-clip/dc] |
| ST-18 | Q:L317, F17 | Denominator G·max_i\|y_i\| per group | `objectives.py:59-61` | partial | H9 [ran HISPO-den, HISPO-dflt] |

### §3.4 Train–inference mismatch (Q:L334-480)

| id | Art. | Claim | Crucible | Status | Evidence |
|---|---|---|---|---|---|
| MM-1 | Q:L338-340, F20 | r_dis at the same θ on the two engines | `mismatch.py` r_dis | done | — |
| MM-2 | Q:L348-352, F21-F22 | Ring: divergence grows at least geometrically | `rollout_probs_diff`, r_dis quantiles per step (`mismatch.py:104`) | partial | logged; no trend alarm |
| MM-3 | Q:L376-380, F23-F24 | TIS: min(r_dis, C), one-sided | `mismatch.py:117-118` | done | [ran TIS-1/2] |
| MM-4 | Q:L384-391, F25-F26 | IcePop: M(k) = k in [α, β], else 0; gate on r_dis, PPO on r_stale; α = 0.5, β = 5; −γD_KL | `mismatch.py:119-122`; `kl_coef` | done | [ran ICEPOP] |
| MM-5 | Q:L395-420, F27-F32 | KPop binary KL; symmetric mask; 4.05e-5 vs 0.4534; approximation 4.047e-5 / 0.174; one hyperparameter φ | `mismatch.py:76-79,123-128` | done | [ran KPOP-1..4] |
| MM-6 | Q:L422 | Kimi: token log-ratio mask in [α, β], sign-blind | `dis` with params | done (≈) | — |
| MM-7 | Q:L422 | Hy3: IcePop [0.5, 4.0]; `rollout_probs_diff` < 0.015 | icepop params; metric | done | — |
| MM-8 | Q:L430-442, F33-F38 | GSPO s_i; sequence-level clip; objective (1/G)Σ min(·) | `mismatch.py:169-176` | partial | H3 [ran GSPO-*] |
| MM-9 | Q:L446, L604 | GSPO clips about 15% of tokens, GRPO 0.13%; ranges 3e-4 / 4e-4 | defaults (`mismatch.py:173`); `clipped_token_share` | partial | For ppo the metric counts every r outside the band whatever the sign of Â (`:114`), so it overstates zero-gradient tokens (P2) |
| MM-10 | Q:L448 | StepFun: GSPO's r_dis diverges on MoE; MIS-PO stays stable | r_dis quantiles per step | done (monitor) | — |
| MM-11 | Q:L454 | DeepSeek: bitwise alignment (batch-invariant kernels, no split-k) | — | missing | out of scope (kernels) |
| MM-12 | Q:L455 | MiMo: QDQ to the MXFP4 grid after every update; Kimi: MXFP4 QAT | `weights.QDQ` (`weights.py:100-161`) | partial | [ran QDQ] idempotent, on the E2M1 grid. No QAT |
| MM-13 | Q:L456, L478 | FP32 LM head, BF16 in, FP32 out (about 2 points of reward) | `hf.py:65-72` | done | [ran FP32HEAD] Δlogits = 0.0 (Qwen3, Llama) |
| MM-14 | Q:L457 | GLM: deterministic topk; frozen DSA indexer | — | n/a | — |
| MM-15 | Q:L463 | R3 routing replay | `routing_replay="r3"` (`trainer.py:470-477`) | done (doc) | vLLM capture open |
| MM-16 | Q:L463 | Top-p candidate-set replay; renormalize over the same set (top-p 0.97 → under 5 candidates) | `sampling.py`; `trainer.py:467-476` | done | [ran TOPP, TOPP2] |
| MM-17 | Q:L464 | DeepSeek: spliced routing replay across checkpoints | routing kept across resumes | done (doc) | #36 |
| MM-18 | Q:L465 | Routing confidence Σ_k | `moe/routing_confidence` (`trainer.py:665-666`) | done | — |
| MM-19 | Q:L480 | Algorithmic corrections need real rollout logprobs and TITO | `valid_for_training`; `chat.py` | done (doc) | — |

### §3.5 Policy optimisation (Q:L482-678)

| id | Art. | Claim | Crucible | Status | Evidence |
|---|---|---|---|---|---|
| AL-1 | Q:L496-503, F39 | Clip bias: π_old = 0.01 → at most 0.012; π_old > 1/(1+ε) = 0.833 can never be upper-clipped | arithmetic | info | [ran PPO-dapo] |
| AL-2 | Q:L505, L533 | DAPO clip-higher: ε_low 0.2, ε_high 0.28 | `mismatch.py:111` defaults | done | [ran PPO-dual] |
| AL-3 | Q:L534 | Dynamic sampling: drop zero-variance groups, refill | `filter_zero_variance` (`scheduler.py:172-178`) | done | `T/test_train_sched.py:85-97` |
| AL-4 | Q:L539-546, F40-F41 | Token-level loss: G = 2, 100 / 10,000 → 5e-3 / 5e-5 vs 9.9e-5 | `aggregate` (`objectives.py:42-47`) | done | [ran DAPO-tok] |
| AL-5 | Q:L555 | Overlong filtering: mask truncated samples (+6 points) | World `LimitPolicy` "mask"; no token-env switch | partial | H11 |
| AL-6 | Q:L557-559, F42 | Soft overlong punishment; L_max 20,480, L_cache 4,096; continuous at 20,480 | `overlong_penalty` (`objectives.py:136-141`) | partial (unwired) | [ran DAPO-soft]; H11 |
| AL-7 | Q:L574 | DAPO: no KL; 16 gradient updates per rollout step | `kl_coef=0`; `updates_per_batch` | done | — |
| AL-8 | Q:L580-589, F43-F44 | CISPO: sg(r̂)·Â·log π; token-level denominator | `mismatch.py:163-168` | done | [ran CISPO-1/2] |
| AL-9 | Q:L584-586, F45 | sg is mandatory: without it the factor is (1 + log π), ≈ −5.9 at p = 1e-3 | — | done | [ran CISPO-sg] no-sg −6.394 = r̂(1 + log π); Crucible r̂·Â |
| AL-10 | Q:L586 | In band, CISPO's gradient equals PPO's; out of band, PPO's is 0 and CISPO's coefficient is held at 1 + ε_high | — | done | [ran CISPO-1/2] |
| AL-11 | Q:L591-593, F46 | Unified PPO/CISPO mask M_it | — | missing | [ran CISPO-mask] (H12e) |
| AL-12 | Q:L597 | M1 bounds [1 − ε_l, 1 + ε_h] vs M2 [0, 1 + ε_h] | `params m2` (`mismatch.py:165`) | done | [ran CISPO-M2] |
| AL-13 | Q:L598 | M2: reward-to-go minus a trajectory baseline; (s_t, a_t) as the unit | `credit_unit=reward_to_go` (`trainer.py:418-436`) | done | [ran MM-rtg] |
| AL-14 | Q:L600 | M2/M2.1: CISPO + MIS + PPO trajectory filtering | one `Correction` only | missing | P2 |
| AL-15 | Q:L604 | GSPO ranges 3e-4 / 4e-4 vs GRPO 0.2 / 0.27 | defaults | done | — |
| AL-16 | Q:L608-610, F47-F48 | SAO objective = DIS; math TIR (0.3, 5.0); coding (0.8, 3.0) | `dis` | done | [ran DIS-*] |
| AL-17 | Q:L614 | TTUR: K = 2 critic updates per policy update | `CriticConfig` (`critic.py:28`) | done | [ran TTUR] |
| AL-18 | Q:L614 | Explained variance | `critic.py:82-84` | done | — |
| AL-19 | Q:L615 | Freeze the critic's attention; train the MoE projections | `HFValue(freeze_attention)` (`critic.py:61-72`) | done | freezes `self_attn` only, so trains a superset |
| AL-20 | Q:L616 | Value pretraining at scale | `pretrain_value` (`critic.py:87-123`) | done | — |
| AL-21 | Q:L617 | λ_policy = 1 − 1/(αl), α = 1.5; λ_critic = 1 | `objectives.py:144-145`; `trainer.py:235-238` | done | [ran VAPO-lam]. l = all trainable tokens of the rollout |
| AL-22 | Q:L619-621, F49 | Skip-observation GAE | `gae` (`objectives.py:148-207`) | done | [ran SAO-skipobs]; latent H12d [ran SAO-stepr] |
| AL-23 | Q:L621 | Token-level value beats step-level (89.8/66.8 vs 85.8/60.5 and 87.3/62.8) | token-level only | done | — |
| AL-24 | Q:L625-634, F50-F51 | MIS-PO trains kept tokens as on-policy: log π·Â, no clip | `mismatch.py:148` | done | [ran MISPO-edge*] gradient = Â |
| AL-25 | Q:L636 | MIS-PO critic: Â = R̂ − V, γ = λ = 1 | `CriticConfig(lam_policy=1.0)` | done | [ran MISPO-crit] |
| AL-26 | Q:L636 | 4 actor : 12 critic minibatches; Muon; lr 2e-6 / 5e-6; unbiased KL 0.001 in the last stage | `updates_per_policy_update`; k3 KL per source | partial | no Muon, no stage schedule |
| AL-27 | Q:L636-638, F52 | Truncation-aware bootstrap R̂ = V(s_T) | `objectives.py:200`; `trainer.py:229,236` | **WRONG** | H1 [ran TRUNC-*] |
| AL-28 | Q:L644 | CompactionRL: token-level normalization (no segment-count bias) | `token_mean_global` + segment items | done (default) | Under `seq_mean*`, each segment is a sample and nothing guards it (P2) |
| AL-29 | Q:L645-647, F53 | Â_{s,i} = (γλ)^{N>s}·A^loc_{s,i}; shared reward at each segment's end | `gae` segments + `shared_terminal` (`objectives.py:190-206`); `trainer.py:231-233` | done | [ran CRL-xgae] scales 0.7351 / 0.8145 / 1. With adaptive λ, the factor at N>s = T/2 is about e^{−1/3} for any T |
| AL-30 | Q:L649, L767 | The summary is trained with the shared reward; no summary-quality reward | `token_credit(train_compaction)` (`credit.py:44`) | done | — |
| AL-31 | Q:L653 | DIS stacks on others (GRPO 84.2/54.8 → 93.5/70.8) | — | info | — |
| AL-32 | Q:L657 | Hy3: clip-higher 0.2/0.28, dual clip c = 10, KL-free, DAPO overlong buffer; estimators gae / grpo / rloo / remax / reinforce++ / gspo / cispo | `ppo_term` (`mismatch.py:82-86`); `ESTIMATORS` (`objectives.py:65`) | done (overlong: AL-6) | [ran PPO-dual] r = 20, A = −1: objective −10, gradient 0; r = 5: −5 |
| AL-33 | Q:L657 | Ring IcePop: KL 0 | default | done | — |
| AL-34 | Q:L659, L762 | Seed shared critic: RM + verifier in [0, 1]; init from the RM; 100-step warm-up; per-task KL 1e-5 / 0 | `HFValue(any LM)`; `kl_coef_by_source` | partial | no two-source critic contract, RM-init helper or warm-up |
| AL-35 | Q:L659 | Seed-Thinking: decoupled GAE | separate λ_policy / λ_critic | done | — |
| AL-36 | Q:L513, L676 | Dr.GRPO no std; Kimi/MiMo mean only; Qwen keeps std | `group_advantages` (`objectives.py:68-110`) | done | [ran DRGRPO]. Population std: [1,0,0,0] → 1.732; unbiased would give 1.5 (P2 note) |
| AL-37 | Q:L670-676 | Aggregations: prompt-mean / seq / token / HisPO constant | `LOSS_AGG` | done | §1 correction (prompt-mean) |
| AL-38 | Q:L507-511 | Single rollout (n = 1) with a critic | critic mode | done (doc) | #16 |

### §3.6 Rewards and credit (Q:L680-784)

| id | Art. | Claim | Crucible | Status | Evidence |
|---|---|---|---|---|---|
| RW-1 | Q:L684 | Grader compute as a scaling axis (12.7%) | `grader/s`, `grader/usd`, `grader/share` | done (doc) | — |
| RW-2 | Q:L700 | MiniMax r = α·r_process + β·r_speed + r_perf; h decreasing; Â = Σ_{p≥t} r_p − B | `speed_bonus` (`objectives.py:123-133`); `_shaped_rewards` (`trainer.py:303-339`) | done (group) / **WRONG** (critic) | [ran MM-rtg]; H4 [ran SPEED, CRIT-step] |
| RW-3 | Q:L701 | Qwen incomplete-trajectory penalty | `LimitPolicy` "penalty:<c>" (`runtime/infra.py:113-125`) | done | In critic mode it is added to the bootstrap (H1) |
| RW-4 | Q:L701 | Tool-format penalty on those tokens only | `tool_format_penalty` (`trainer.py:440-449`) | done | [ran QWEN-fmt] |
| RW-5 | Q:L713 | DeepSeek: the actor is the GRM; RL on the GRM | — | missing | A2-27 |
| RW-6 | Q:L721 | GRS rubric built offline, grounded in the task | `check_rubric_grounding` (`credit/gar.py:75-83`) | partial | no rubric-building agent |
| RW-7 | Q:L723-725, F54 | R = R_test·S_sol·S_beh | `group_ops.grs` (`:61-80`); `credit/gar.grs` | done | [ran GRS] |
| RW-8 | Q:L729-731 | Mixed-group grader; hack → 0, then recompute group stats | `zero_hacks` + `gar` | done | [ran GAR-hack] |
| RW-9 | Q:L733-735, F55 | GAR λ = Σ_P A / Σ_P fA; Σ_P conserved; λ cap; centring; async with fallback | `group_ops.gar` ✓; `credit/gar.gar` ✗ | partial | H8 [ran GAR-*]; fallback `runtime.py:382-412` |
| RW-10 | Q:L744 | Kimi tournament GRM; judge protocol | `tournament` (`group_ops.py:136-157`) | partial | no protocol |
| RW-11 | Q:L744 | Kimi verbosity σ·ℓ₀ auto-lose | `verbosity_group_fn` | **WRONG** (deliberate) | H10 |
| RW-12 | Q:L748 | GLM: rule + ORM + GRM mix; human style anchors | — | partial | A2-23 |
| RW-13 | Q:L752 | StepFun pairwise vs a fixed reference; confidence → BT win rate; O(n) | `bt_pairwise` (`group_ops.py:119-133`) | done | — |
| RW-14 | Q:L752 | Confidence length penalty; fabrication → 0; MetaRM; ternary → asymmetric binary | — | missing | A2-20 |
| RW-15 | Q:L756 | Ling bidirectional focus (per-dimension saturation shifts weight) | — | missing | A2-22 |
| RW-16 | Q:L758, L833 | Soft hacks (compatibility branches, broad exports, swallowed errors) | — | missing | A2-34 |
| RW-17 | Q:L768 | DeepSeek DAG critical-path latency | `dag_critical_path` (`credit.py:96-107`) | done | [ran DAG, DAG-chain] |
| RW-18 | Q:L769 | PARL: subagents frozen and excluded | `frozen_actors` (`credit.py:44`; `trainer.py:465`) | done | — |
| RW-19 | Q:L769 | r_parallel + r_finish + r_perf with λ1, λ2 annealed to 0 | — | missing | grep "anneal" finds the lr only |
| RW-20 | Q:L769-771, F56 | Critical steps Σ(S_main + max S_sub) | `critical_steps` (`credit.py:77-93`) | done | [ran PARL-cs] 6 / 6 / 7. Edge: sub calls before the first main call are dropped [ran PARL-pre] |
| RW-21 | Q:L784 | Pipeline: verifier → rubric → hack zero → redistribution → shaping → GAE / broadcast | group stage, then trainer | partial | critic mode skips shaping (H4) |

### §3.7 Reward hacking (Q:L786-837)

| id | Art. | Claim | Crucible | Status | Evidence |
|---|---|---|---|---|---|
| HK-1 | Q:L808 | Qwen: intercept any call with repo link + git/curl/wget; explicit refusal | `REMOTE_FETCH` (`intercept.py:50-55`) | partial | H7 [ran INTERCEPT] |
| HK-2 | Q:L809 | Ling: drop `--all` from `git log`; 0.2% cheat | guard `git_scrub` | partial (not probed) | — |
| HK-3 | Q:L813 | DeepSeek: AppArmor + eBPF; environment crash = failure + recoil | `IsolationSpec`; `agent_broke_env` = 0 | partial | A2-35, A2-39 |
| HK-4 | Q:L817 | LongCat: Lean4 AST, 9 patterns, 97.9% → 27.9%; pass-rate jump as the signal | `PassRateAlarm` (`monitors.py:37-49`) | partial | alarm done ([ran] in `T/test_train_sched.py:109-118`); Lean AST missing (out of enterprise scope) |
| HK-5 | Q:L821 | Kimi: build failure or fake artefact → 0; kernel cheats; public + hidden verifiers | verify perf patterns | partial | — |
| HK-6 | Q:L825 | GLM: rules for recall, LLM judge for precision; intercept and return dummy info; rollout continues | `intercept` action "dummy" (`world.py:455`); `hack_detector` | done | GLM's sample paths (`/workspace/.eval/…`, `*hidden*`) are not default rules [ran: miss] |
| HK-7 | Q:L829 | MiMo hack agent until clean; confirmed hacks < 2% | A13 static; alarm at 2% (`monitors.py:90-94`) | partial | — |
| HK-8 | Q:L837 | Metrics: confirmed hack rate, AST pass rate, `rollout_probs_diff` | logged | done (no AST) | — |

### §3.8 Noise and context (Q:L839-918)

| id | Art. | Claim | Crucible | Status | Evidence |
|---|---|---|---|---|---|
| NZ-1 | Q:L869 | MiMo: 8 consistent reruns | `admission.screen(k=8)` (`admission.py:89`) | done | [ran RERUN-k8] |
| NZ-2 | Q:L869 | Infra failures masked from the loss | `mask_sample`; None excluded | done | [ran MASKADV] |
| NZ-3 | Q:L870-871 | Ling: unstable instances removed; StepFun: transient failures masked | screen; infra mask | done | — |
| NZ-4 | Q:L876-878 | microVM pause / fork / snapshot; DSec substrates; ASandbox 100 ms | — | partial / missing (doc) | A2-41/42 |
| NZ-5 | Q:L888 | LongCat noise curriculum on the clean–noisy gap | `NoiseCurriculum` (`noise.py`) | done | [ran NOISE-cur] |
| NZ-6 | Q:L892 | GLM keep-recent k = 5 | `strategy="keep_recent"`, default k = 3 (`policy.py:89-99`) | done (configurable) | — |
| NZ-7 | Q:L892 | GLM tiered HCM: above T = 32k, discard all tool history | `"tiered"`: shrink old tool outputs, then summarize (`policy.py:97-98`) | partial | different rule |
| NZ-8 | Q:L893 | LongCat hybrid: summary at 80K + discard-all at max turns + progressive thresholds | `"hybrid"`: N summaries, then discard-all | partial | triggered by summary count, not turns |
| NZ-9 | Q:L894 | Kimi: context management per task group | per-harness `ContextConfig` | done | — |
| NZ-10 | Q:L895, L915 | Interleaved / preserved thinking; StepFun selective retention | thinking modes | partial | selective mode missing |
| NZ-11 | Q:L899 | CompactionRL: compact when < 10,240 tokens remain; keep k = 2; summary in the objective | default 0.8 × 128k (25,600 left), keep 3 | partial | [ran CTX-dflt] configurable; the trigger is proportional, not absolute |
| NZ-12 | Q:L900 | MiniMax: context management as an action | `mode="action"` (`policy.py:91-94`) | done (doc) | — |

### §4 table rows not covered above (Q:L924-936)

| id | Claim | Status |
|---|---|---|
| TB-1 | DeepSeek effort-conditioned length penalty; effort-homogeneous groups | missing (A2-60) |
| TB-2 | Kimi per-token regularization τ for extreme off-policy | missing. k3 KL to a reference is a different thing |
| TB-3 | MiniMax Prefix Tree Merging, 40× | done (`prefix_sharing`, #50) |
| TB-4 | Ling zlib repetition penalty; MiMo ToolRep | missing (A2-61). The zlib hits in `CR` are unrelated |
| TB-5 | Hy3: freeze the MoE expert bias | missing (no hit) |
| TB-6 | GLM: evict failed environments + group backfill | done (`scheduler.py:130-150`) |
| TB-7 | MiMo per-source expiry | done (`max_staleness_by_source`) |
| TB-8 | Seed: small KL against hacking; dual-track verifiers | partial |

§5 has six routes, all represented: route ④ OPD is m13 C-5. §6 lists open questions with no testable numbers beyond MiMo 4 and StepFun about 20.

### Other articles (training content only)

| id | Source | Claim | Status | Evidence |
|---|---|---|---|---|
| Z-7 | Z | IS covers drift and engine gap; low-bit rollout only inside the clip | done | corrections |
| Z-9 | Z | Per-action and per-trajectory rewards | partial | critic mode drops per-action rewards (H4) |
| Z-10 | Z | Discard when current − version > staleness; typical 1–2 | done | ST-1 |
| Z-11 | Z | Bounded input queue = max concurrent rollouts | done | ST-4 |
| Z-12 | Z | An interrupted task goes back to the head of the queue | missing | A-1 |
| MH-1 | MH:L195-220 | Exact sampled ids + behaviour logprobs; never re-encode | done | TITO |
| MH-2 | MH:L207-211 | Recomputing with current weights makes the ratio 1 | done | NEEDS_OLD refusal; DIS |
| MH-3 | MH:L545-558 | Root–leaf branches as sequences; a misaligned turn stays as context | partial | Crucible masks the whole rollout (stricter) |
| MH-4 | MH:L704-709 | One score; None when the verifier never ran | done | — |
| MH-5 | MH:L805 | One harness per GRPO group | partial (doc) | #43 |
| MH-6 | MH:L806 | Sampling temperature = recompute temperature | **missing** | H5 |
| MH-7 | MH:L878-899 | Efficiency bonus ≤ 0.1, success-gated | done | `speed_gate`, `with_speed` |
| MH-8 | MH:L1194 | 35–58% of steps without contrast | done | dynamic sampling + metric |
| CH-1 | `2605.09998.txt:2067-2071` | Offline GRPO with groups formed per *state* | missing (P2) | needs World fork + a state group key |

---

## 3. Probes

Every row ran. "FAIL" means Crucible deviates from the article.

| Probe | Article | Result |
|---|---|---|
| KPOP-1..4 | F30-F32 | PASS: 4.0473e-5; 0.4534; 4.047e-5 / 0.174; mask [T, F] at φ ∈ {1e-3, 1e-2, 1e-1}; masked-token gradient 0 |
| ICEPOP ×6 | F25-F26 | PASS: gradient = r_dis·r_stale·Â in band (0.77; −3.8; 5.0), 0 outside. The r = 0.5 edge "FAIL" is float; exact re-probe passes |
| TIS-1/2 | F24 | PASS: r_dis = 0.01 → gradient 0.01 (one-sided); r_dis = 5 → 2 |
| DIS-coding / DIS-math-TIR | F8-F9, Q:L610 | PASS: open bands (0.2, 4) and (0.7, 6); gradient r·Â; sign-blind |
| MIMO4-1/2/3 | F10-F12, Q:L268 | PASS (edge float re-probed); entropy low → pos [0.182, 5.5], neg [0.22, 4.545]; high → reversed |
| MISPO-T100/1000/5000, edge×3, rej, asym, tok, src | F13-F15, Q:L284-288 | PASS: 0.670 / 0.0183 / 2.06e-9. src: rollout as π_θold → filters nothing (gap) |
| HISPO-g×3, clip, dc | F16-F19 | PASS |
| HISPO-den | F17 | **FAIL** 0.550 vs 0.775 |
| CISPO-1/2/sg/M2 | F43-F45, Q:L586,597 | PASS |
| CISPO-mask | F46 | **FAIL**: not implemented |
| GSPO-seq_ratio / GSPO-clip | F35-F38 | PASS |
| GSPO-token_mean_global | F36 | **FAIL**: value −0.05 vs 0.25; gradient ratio 1.4 |
| PPO-dual / PPO-dapo | Q:L657, F39 | PASS |
| DAPO-tok / DAPO-soft | F40-F42 | PASS |
| SAO-skipobs | F49 | PASS |
| SAO-stepr | F49 | **FAIL** −0.0635 = −0.5(1 − γλ) (latent) |
| CRL-xgae / CRL-critic | F53 | PASS |
| TRUNC-R0 | F52 | PASS (R = 0 only) |
| TRUNC-R1 / TRUNC-R-1 / TRUNC-tr1.0 | F52 | **FAIL** 1.5 / −0.5 / target 1.5 with advantage +1.0 |
| MISPO-crit / VAPO-lam / DRGRPO / MASKADV | Q:L636, L617, L513 | PASS |
| GLM-drop | F5, Q:L227 | PASS |
| RING-s0 | Q:L231 | **FAIL** σ = 0 → staleness 1 |
| RING-s1/s2 | Q:L231 | PASS |
| LING-S1/2/4 | F6 | PASS cap. Staleness 1 / 5 / 10 contradicts `scheduler.py:15-16` |
| WFIFO-test | — | vacuous: 0 against a bound of 10 |
| WFIFO-sweep | Q:L190 | PASS: lag 30 → 1 |
| OFFPOL, OFFPOL-extra | Q:L272 | share cap holds. **τ overrun 8 at τ = 3; disaggregated stall at step 1** |
| WAIT-f* | — | `wait` alone bounds τ = 2. With the cap the run trains 1 step |
| KIMI-λ / C3PO-Φ | Q:L168,178 | run |
| GAR-ops / GAR-cap / GAR-mixed / GAR-allpass / GAR-hack | F55 | PASS |
| GAR-cr0.5 / GAR-crcap | F55 | **FAIL** (H8) |
| GRS | F54 | PASS |
| KIMI-verb / KIMI-verb2 | Q:L744 | **FAIL** (H10) |
| KIMI-tour | Q:L744 | gap shown |
| PARL-cs / DAG / DAG-chain | F56, Q:L768 | PASS |
| PARL-pre | F56 | FAIL (edge, P2) |
| INTERCEPT ×8 | Q:L808, L825 | 4 PASS; **4 miss**: two `git remote add` variants, two GLM sample commands |
| TTUR / QWEN-fmt / MM-rtg | Q:L614, L701, L700 | PASS |
| CRIT-step / SPEED | Q:L700, Z-9 | **FAIL** (H4) |
| GSPO-dflt / HISPO-dflt | — | **FAIL** (config default) |
| MIMO-pm | Q:L268 vs MiMo code | "FAIL" against the article's wording; matches the MiMo fork (no defect) |
| TEMP-guard | MH:L806 | **no guard** (H5) |
| FP32HEAD | Q:L456 | PASS, Δ = 0.0 |
| QDQ / TOPP / TOPP2 / RERUN-k8 / NOISE-cur | Q:L455, L463, L869, L888 | PASS |
| CTX-dflt | Q:L899 | defaults ≠ CompactionRL (configurable) |

---

## 4. Tests that check Crucible against itself, or check nothing

| Test | Problem | Sev |
|---|---|---|
| `T/test_train_sched.py:82` | lag ≤ 10 where the actual is 0 (S = 1); the article never states the bound | P1 |
| `T/test_train_sched.py:106` | `≤ max(1, len(b)//4)*4` = 16 of 16: always true; hides H2 | P1 |
| `T/test_train_critic.py:53-61`, `T/test_train_math.py:140-141` | truncation only at R = 0; cannot tell add from replace (H1) | P1 |
| `T/test_train_critic.py:24-44` | trainer advantages compared with Crucible's own `gae`. Acceptable only because `gae` has independent goldens, but those share the R = 0 blind spot | P2 |
| `T/test_train_math.py:77-84` | GSPO at s = 1 only; no aggregation; no gradient | P1 |
| `T/test_train_math.py:61-66` | HisPO −1.28 is Crucible's own reading of the three bounds (the article gives no number) | P2 |
| `T/test_group_stage.py:20-25` | credit/gar at lam = 0.5 asserts its own interpolation as "λ caps the move", not the article's λ | P2 |
| `T/test_shaping.py:88-95` | asserts the bounded verbosity rule, not the article's | P2 |

The discriminating replacements are the probes above: OFFPOL-extra, WFIFO-sweep, TRUNC-tr1.0, GSPO-* against the reference, GAR-cr*.

---

## 5. Picks

Verdicts:
- **DEPEND**: call their code.
- **PORT**: copy (the licence permits).
- **PATTERN**: rebuild the idea.

Source paths are under `RL-Environments/06-frameworks-and-env-code/`.

| # | Gap (sev) | Verdict | Effort | Source / what to build |
|---|---|---|---|---|
| 1 | H1 truncation bootstrap (P0) | PORT | S | AReaL semantics (`infra-inclusionai-areal/tests/test_ppo_actor_truncation.py:282-300`): a truncated rollout's task reward → 0, bootstrap from its last valid token. Keep `bootstrap_value = v[-1]` (AReaL uses the same position). Test with R ∈ {1, −1, penalty}. Decide whether a World `max_steps` stop the verifier scored is "truncated" (bootstrap) or terminal (keep R, no bootstrap). Never both. |
| 2 | H2 `max_offpolicy_frac` (P0/P1) | PATTERN | S–M | Exempt groups at their bound from the cap: train them, or evict them as stale. Reserve admission room for on-policy work, or relax the cap while it blocks (AReaL `max_head_offpolicyness` admission, `infra-inclusionai-areal/tests/test_rollout_controller.py:226-373`). Regression: the sim in both topologies, asserting steps == requested and staleness ≤ τ. |
| 3 | H3 + H9 loss aggregation per correction (P1) | PATTERN | S | Default `loss_agg` from the correction (gspo → seq_ratio, hispo → const_max_len). Refuse a mismatch unless overridden, as MiMo's verl refuses prompt-mean with a non-vanilla loss (`xiaomimimo-verl/verl/workers/config/actor.py:218-227`). Compute HisPO's denominator per group (pass group ids into `aggregate`). |
| 4 | H4 critic-mode shaping (P1) | PATTERN | S | Build the critic's reward vector from `_shaped_rewards` plus `Turn.step_reward` at each call's last token, credited to the action that earned it. This also fixes the latent carry-forward in `gae` (H12d). |
| 5 | H5 temperature guard (P1) | PATTERN | S | Refuse an item whose `Turn.sampling["temperature"]` ≠ the trainer's, or recompute at the item's own temperature (MH:L806). |
| 6 | H7 interception (P1) | PATTERN | S | The Qwen rule: the task's repo slug + a network keyword (git / curl / wget), covering `git remote add\|set-url` and later `fetch/pull <remote>`. Add the GLM sample patterns as a configurable rule set. Keep the egress guard as the backstop. |
| 7 | H6 vacuous tests (P1) | PATTERN | S | Promote the probes: window lag at S > 1 against greedy; per-group off-policy share plus τ under the cap; truncation at R ≠ 0; GSPO gradients against a reference at s ≠ 1. |
| 8 | H8 two GARs (P2) | PATTERN | S | One operator with the article's λ (normalizer + cap + centring), i.e. `group_ops.gar`. Make `gar_group_fn` use it; keep the interpolation as an explicit `blend`. |
| 9 | H10 Kimi verbosity (P2) | PATTERN | S | An article mode inside `tournament`/`bt_pairwise`: over σ·ℓ₀ loses, with ℓ₀ from a cold-start reference policy. Keep the bounded penalty as an alternative. Fix `M12-train.md:35`. |
| 10 | H11 DAPO overlong (P2) | PORT | S | verl's DAPO reward manager `overlong_buffer` (`infra-volcengine-verl/verl/experimental/reward_loop/reward_manager/dapo.py:33-39`) as a reward wrapper. Add an overlong-filter mask switch for `finish == "length"`. |
| 11 | CISPO unified mask + composition with MIS (P2) | PORT | S | slime `compute_cispo_loss` (`infra-thudm-slime/slime/utils/ppo_utils.py:324`). Allow a correction stack (gate ∘ weight). |
| 12 | Sequence-level gates per trajectory (P2) | PATTERN | M | Pass rollout ids into `correct`; compute ρ̄, HisPO's sequence gate and GSPO's s_i over all of a rollout's rows, whatever the item unit. |
| 13 | MIS-PO with `old_logprob_source="rollout"` (P2) | PATTERN | S | Refuse it: x ≡ 1, so the filter does nothing. |
| 14 | Ring σ semantics (P2) | PATTERN | S | Count cuts for generating rollouts only, not `grading`. Document that σ needs τ to bound staleness. Fix the `scheduler.py:15-16` claim. |
| 15 | Prioritized resume of cut rollouts (P2) | PATTERN | S | AReaL re-queues at the head (Z-12); Kimi resumes them first. |
| 16 | Context presets (P2) | PATTERN | S | CompactionRL preset (absolute 10,240 remaining, keep 2), GLM k = 5, HCM "discard tool history at T", LongCat max-turn trigger. |
| 17 | Missing reward features (P2): Ling focus, StepFun asymmetric mapping + confidence length penalty + MetaRM, PARL λ annealing, Kimi per-token τ regularizer, Hy3 expert-bias freeze, zlib repetition, state-grouped GRPO | PATTERN | M each | Defer until a pack needs them. |
| 18 | Lean4 AST validator (P2) | — | L | Out of enterprise scope; the pass-rate alarm covers the signal. |
