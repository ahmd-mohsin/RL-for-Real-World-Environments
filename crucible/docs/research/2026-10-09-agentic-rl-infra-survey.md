# Notes: "Agentic RL Infra: a survey of mainstream technical routes" (Wei Ziyang, 2026)

- **Source:** http://qingkeai.online/blog/2026-Agentic-rl-infra (mirror of weiziyang.wiki/posts/post-training-rl/agentic-rl-infra-technical-roadmap-overview), in Chinese.
- **Local copy:** `RL-Environments/07-articles/qingkeai-2026-agentic-rl-infra.md` (+ `raw/`, `.links.txt`).
- **Scope:** 11 public 2025–2026 technical reports: Xiaomi MiMo-V2.6, DeepSeek-V4/V4.1, Moonshot Kimi K2.5/K3, Zhipu GLM-5/5.2, Alibaba Qwen (GSPO, Qwen3-Coder-Next, Qwen3.5), MiniMax M1/M2.x (Forge, CISPO), StepFun Step 3.5 (MIS-PO), Meituan LongCat (DORA, HisPO), Ant Ling/Ring (ASystem, IcePop, KPop, C3PO++), ByteDance Seed, Tencent Hunyuan.
- **Read:** 2026-10-09, in full.

## 1. The thesis in one paragraph

Agentic RL infrastructure has largely converged. The central problem is **keeping trainer GPUs busy while the gradient signal stays trustworthy, inside a long-tailed, noisy external environment**: every design trades throughput against off-policy-ness.

DeepSeek-V4.1 states that **engineering data and environment pipelines currently has a higher marginal return than algorithmic innovation in post-training**. MiMo treats **grader compute** as a scaling axis of its own (12.7% of Pro's cost), alongside batch size and environment diversity. Both are independent confirmations of Crucible's premise: environments, verifiers and dense credit are the bottleneck, not the optimizer.

## 2. The four challenges and four planes

**Challenges:**
1. Trajectories got long (Kimi K3 RL at 1M context; Qwen3-Coder-Next SWE RL went from 50 to 130 turns on average).
2. Environments got messy (sandboxes, browsers, MCP tools, terminals, each with its own latency and failure rate).
3. Rewards got sparse (hundreds of steps, one binary outcome).
4. The tail is heavy: about 5% of samples take about 80% of generation cost (StepFun); MiMo measured a 90× spread in mean generated tokens and a 66× spread in active time across 25 data sources.

**Planes:**
- **Control plane:** scheduling, versions, recovery, observability.
- **Data plane:** trajectories, logprobs and rewards moving between rollout, queue, verifier and trainer.
- **State plane:** KV cache, environment state, tool sessions, partial rollouts.
- **Weight plane:** trainer → rollout weight sync and version switching.

## 3. What each route does (compressed)

### 3.1 Colocated vs disaggregated
- **Colocated:** Kimi K3 (offload weights and optimizer to NVMe, a CPU KV pool), Hunyuan, DeepSeek-V4.1, and Ant ASystem (P2P weight swap of 1T parameters in 10 s). MiMo appears colocated too.
- **Disaggregated:** GLM-5 (a rollout orchestrator with per-task microservices, 1k+ concurrent rollouts), MiniMax Forge (Gateway + Data Pool middleware that plugs in any white- or black-box scaffold), LongCat DORA (32k concurrent environments on about 400 machines), Qwen3.5 (3–5× end to end).
- **Consensus:** **environment execution leaves the GPU pool** and scales on CPU (DeepSeek DSec, Ling ASandbox, LongCat).
- **Common stack:** Megatron-LM + vLLM/SGLang + Ray, with an in-house agentic orchestration layer on top (slime, Forge, DORA, ASystem, Steptron, MegaFlow).

### 3.2 Sync vs async; long tail
- **Partial rollout on colocated GPUs:** cut in-flight sequences when the batch is full and resume them under the new weights.
  - Kimi uses a completion ratio λ.
  - MiMo uses staleness = 4 and pays re-prefill.
  - DeepSeek-V4.1 pauses at any token boundary and resumes from persistent state with no re-prefill. It schedules **per sample** (it tried batch-level scheduling, which made metrics oscillate, and prompt-level, which got stuck on group tails).
  - Ring C3PO++ uses a global token budget per round.
- **Fully streaming async on disaggregated GPUs:** GLM (push weights every K steps, reset the optimizer after each update), LongCat (2–4×), StepFun search agent (10×).
- **Windowed FIFO (MiniMax):** the trainer takes only finished trajectories inside a sliding submission window W = 0.3N, which bounds the maximum off-policy lag.
- **Length bias:** short samples finish first, so early batches skew short. DeepSeek limits concurrency per dataset and drops early short returns.

### 3.3 Staleness control
- **Version cap + drop:** GLM drops a trajectory when w′ − w₀ > τ, using the oldest version that touched it. Ring drops after more than σ cuts.
- **Admission gating:** Ling caps the in-flight pool at about S × B, so staleness ≤ S steps, combined with sub-second weight broadcast.
- **Algorithmic masking:**
  - **GLM DIS:** the ratio is π_θ / π_rollout (no π_old); outside [1−ε_l, 1+ε_h] the token gets zero weight, on both sides.
  - **MiMo:** four bounds by advantage sign, initially [0.2, 5], adapted to entropy.
  - **StepFun MIS-PO:** binary filtering. Token ratio must be in [0.5, 2] and the trajectory's geometric-mean ratio in [0.996, 1.001]; kept tokens have weight 1. Stable up to about 20 steps of staleness.
  - **LongCat HisPO:** decompose r = r_dis × r_stale; gate on r_dis (sequence then token), then clip on the full ratio with three bounds.

### 3.4 Train–inference mismatch
- **Why it matters:** the same weights give different logprobs in the two engines, and MoE expert choice flips. GSPO measured about 10% of activated experts changing after one update. Ring proves the mismatch grows at least geometrically.
- **Algorithmic remedies:**
  - **TIS:** truncated π_train/π_infer; the de facto baseline in slime, verl and OpenRLHF.
  - **IcePop:** mask outside [0.5, 5].
  - **KPop:** binary-KL mask, because ratio thresholds over-mask low-probability tokens. SWE-bench Verified 70.8 → 76.3.
  - **GSPO:** a sequence-level ratio; no routing replay needed. StepFun found its mismatch keeps diverging on MoE.
- **System remedies:**
  - DeepSeek: bitwise-aligned kernels and native FP4.
  - MiMo QDQ and Kimi MXFP4 QAT: shared quantization in both engines.
  - **FP32 LM head:** MiniMax and Ling found it independently; Ling measured about +2 reward points. The cheapest single fix.
  - GLM: deterministic top-k.
- **Routing replay:** MiMo R3 (also replays the top-p candidate set), DeepSeek spliced replay across checkpoints. StepFun's **Routing Confidence Σ_k** tells you whether replay is needed.
- **The prerequisite all of these share:** **token-in-token-out with rollout logprobs recorded.** A detokenize–retokenize round trip breaks token boundaries.

### 3.5 Policy-optimization algorithm
- **GRPO's four pathologies:**
  1. Sequence reward with token-level importance sampling.
  2. Clipping systematically removes low-probability tokens (e.g. "wait"; DAPO's clip-higher fixes it).
  3. Groups clash with async and with single-feedback environments (GLM SAO: group size 1 plus a critic).
  4. No critic means coarse credit.
- **DAPO ablation** (AIME24 on Qwen2.5-32B):

  | Step | Score |
  |---|---|
  | Naive GRPO | 30 |
  | + Overlong filtering | 36 |
  | + Clip-higher | 38 |
  | + Soft overlong punishment | 41 |
  | + Token-level loss | 42 |
  | + Dynamic sampling (full DAPO) | 50 |

  The largest single gains are overlong filtering (+6) and dynamic sampling (+8).
- **CISPO:** clip the IS weight (stop-gradient) so every token keeps its gradient.
- **GSPO:** sequence ratio; clips 15% of tokens vs GRPO's 0.13%, yet trains more efficiently.
- **SAO:** two critic updates per policy update, frozen critic attention, value pretraining, length-adaptive GAE, and **skip-observation GAE** (observation tokens removed from the Bellman chain). Token-level GAE beats step-level.
- **CompactionRL:** token-level loss normalization and cross-trajectory GAE over compaction segments.
- **The critics' price:** every team that went back to a critic paid three or four extra designs to make the value model train.

### 3.6 Sparse reward and credit assignment
- **Outcome + shaping:**
  - MiniMax: r = α·process + β·speed + perf, where speed = h(T_completion / T_baseline) cut SWE-bench wall time by 37%. Advantage = reward-to-go minus a trajectory baseline, with the atomic unit a single (s_t, a_t).
  - Qwen: an unfinished-trajectory penalty plus token-level penalties on malformed tool calls.
- **GRM / rubric / tournament:**
  - DeepSeek: the actor is its own GRM, also trained with RL.
  - **MiMo GRS:** an offline rubric built by an agent comparing several rollouts. The rubric is grounded in the task, not the samples. Reward = R_test × S_sol × S_beh, so a failure stays 0 and passing solutions get ranked.
  - **MiMo GAR:** a groupwise agentic grader compares the passing patches of a mixed group on five axes: approach, precision, minimality, no side effects outside the task's scope, and craftsmanship. It gives each a quality factor, redistributes positive advantage among passing solutions with the **total positive mass conserved**, zeroes confirmed shortcut solutions and recomputes the group statistics. Without GAR, turns and tokens grew and the pass rate stalled; with it, turns stayed stable.
  - Kimi: a tournament GRM with rubrics generated per task, and a verbosity budget (candidates longer than σ·ℓ₀ lose automatically).
  - GLM: rules + ORM + GRM mixed, with human-written answers as style anchors.
  - StepFun: a pairwise GenRM against a fixed reference (O(n) comparisons, Bradley–Terry win rate) plus a MetaRM that penalizes "right answer, wrong reasoning".
  - Ling: a bidirectional RM that moves weight away from rubric dimensions that are already saturated.
- **Critic credit:** Seed's shared critic, initialized from the RM; SAO's skip-observation GAE.
- **Structural credit:**
  - CompactionRL: summaries share the task reward; cross-trajectory GAE.
  - DeepSeek: the critical path of a multi-agent event DAG as the latency penalty.
  - Kimi PARL: subagents frozen and only the orchestrator trained; **critical steps** = main-agent steps + the slowest subagent per stage. Auxiliary rewards annealed to 0.
- **Open question:** fine-grained *outcome* grading (more grader compute) is consensus; *process* supervision is not. Most teams still broadcast a sequence-level advantage.

### 3.7 Reward hacking (defensive lessons only)
- **Layers:** environment hardening, rule plus LLM-judge detection, verifier hardening (AST or rule checks instead of loose ones), and adversarial screening with reward zeroing.
- **Lessons that matter to us:**
  - Agents will re-acquire answer sources creatively. Interface narrowing and monitoring beat simple deletion.
  - **A sudden jump in training pass rate is itself a signature of a hack.** LongCat's verifier fix exposed about 70 points of fake progress.
  - **Environment failures caused by the agent count as failures, not infra noise** (DeepSeek reports such crashes as failures with a negative signal).
  - For perf tasks, penalize speed that isn't real computation: replaying recorded work, caching inputs across repetitions, silently lowering precision (Kimi, kernel tasks).
  - GLM-5.2 **intercepts a detected shortcut and returns harmless feedback instead of aborting the rollout**, because aborting destabilized async training.
  - MiMo runs a dedicated red-team agent **until it finds nothing in any environment**, then audits during training; confirmed hacks stay below 2%.
  - **"Soft hacks":** tests pass but quality degrades (speculative compatibility branches, overly broad exports, swallowed exceptions, relaxed validation). Binary verifiers cannot see this; group comparison can.
- **Consensus:** an arms race. It needs a harden → audit → harden loop with quantified monitors: confirmed-hack rate, AST pass rate, `rollout_probs_diff`.

### 3.8 Environment noise and long-horizon context
- **Noise:**
  - MiMo admits a task only after **8 consecutive consistent reruns**, and masks infra failures out of the loss.
  - Ling drops instances whose gold patch ever flakes.
  - LongCat **injects instruction and tool noise as a curriculum** (τ²-Noise 62.2 → 67.1).
- **Sandboxes:**
  - Kimi AgentENV: Firecracker microVMs with pause, **fork (side-effect-free reward judging)** and snapshot; 51M sandboxes.
  - DeepSeek DSec: four backends, millions of concurrent sandboxes.
  - Ling ASandbox: 100 ms start.
- **Synthesis:**
  - Qwen: 808k real-PR instances plus 852k synthetic bugs, with QA-agent and verifier-validity denoising.
  - GLM: 10k+ SWE environments.
  - StepFun: 50k verified environments, and **environment building is itself trained with RL** (about 40% success).
  - Ling: about 220k Docker images produced by Claude Code + MCP.
- **Context management:**
  - Heuristics at inference time (keep-recent-k, discard-all, summary).
  - **Context management as an RL action:** MiniMax folds it into the environment dynamics; CompactionRL compacts inside RL (SWE-bench 59.8 → 66.8).
  - Kimi varies the context mechanism per task group so the policy doesn't overfit one harness.
  - Preserving vs dropping thinking history matters for 100+ turn sessions.

## 4. Six routes (the author's synthesis)
1. Sync or near-sync colocated, with masking.
2. Async + partial rollout + staleness control (orthogonal to colocated vs disaggregated).
3. Sequence-level IS (GSPO) instead of routing replay.
4. Specialize, then distill (multi-teacher on-policy distillation replacing parts of RL).
5. Critic vs critic-free.
6. **Environment as infrastructure.**

Still open:
- colocated vs disaggregated;
- staleness caps;
- eliminating vs correcting mismatch;
- routing replay;
- the critic's value;
- process reward;
- the hacking arms race;
- environment reproducibility;
- **long-horizon credit assignment** (all current answers are approximations without bias bounds).

## 5. What this means for Crucible

Crucible is the environment, verifier, credit and benchmark layer; the trainer is M12. Most of the survey is trainer-side, but the requirements it puts on environments and rewards are direct.

| Survey finding | Crucible today | Gap → action |
|---|---|---|
| Environment and grader compute are the scaling axes (DeepSeek, MiMo) | The thesis; M1/M7/M8 | none; cite as external evidence in DESIGN §1 |
| Token-in-token-out with rollout logprobs recorded is the prerequisite for every mismatch fix | The capture proxy records messages; logprobs not guaranteed | **M10/M11:** rollout records must carry token ids, behaviour logprobs and the policy version per span (GLM's w₀ rule needs the oldest version) |
| Agent-caused environment failure is a failure, not infra | C6 masks infra failures; M7 counts candidate crashes as fail | **M1/M6:** attribute environment crashes. If a crash follows a policy action that broke the environment, use failure_kind `agent_broke_env` → reward 0 plus a negative signal, never masked |
| Truncation ≠ failure (DAPO overlong filtering +6; StepFun value bootstrap) | `max_steps` episodes are scored as failures | **M6:** `truncated` episodes either mask or bootstrap. Our κ(1−O)Φ term is a natural bootstrap: a truncated run earns its Φ, and that choice is recorded |
| GAR: redistribute advantage among passing solutions, mass conserved; zero confirmed hacks and recompute group stats | `compose.turn_advantages` conserves mass; rubric bonus is additive (α·rubric) | **M6:** a GAR-style option driven by validated rubric items (C4) and an M7 hack verdict; recompute group stats after zeroing |
| Multiplicative rubric anchoring R_test × S_sol × S_beh (MiMo GRS) | R = G·[O(1+α·rubric) + κ(1−O)Φ] | equivalent on failures (O = 0 ⇒ no rubric credit); keep ours, note the equivalence |
| Soft hacks (swallowed exceptions, broad exports, relaxed validation) pass binary tests | A3 negatives, collateral invariants, the M7 static scan | **M8 Adversary:** add soft-hack mutants as negatives; **M6 C4:** a minimality/scope rubric item |
| Perf tasks: speed that isn't real computation (input caching across reps, replay, lower precision) | Fresh HOME, cohort kill, outside timing, bit-exact checks; task04 perf reuses identical inputs per round (B.10 open) | **task04 re-versioning:** vary perf inputs per round; keep the replay/precision checks in the A13 suite |
| A sudden training pass-rate jump signals a hack | none (M12 not built) | **M12/M13 monitor:** alarm on pass-rate jumps; re-audit with the full tier (`tiers.audit_sample`) |
| Intercept a detected shortcut and return harmless feedback; don't abort | P27 masks the episode on exposure; the guard blocks egress with a refusal | consistent: blocked probes already get a refusal and the rollout continues. Keep masking only for *exposure* (the shortcut worked) |
| Red-team until nothing is found in any environment (MiMo) | A13 suite, Adversary role | **M9:** run the attack suite plus Adversary cheats per task until clean; record the confirmed-hack rate on the card |
| Flaky admission: 8 consistent reruns (MiMo); drop flaky gold (Ling) | A4: oracle false-negative rate 0 over ≥ 40 draws | ours is stricter; keep |
| Noise injection curriculum (LongCat) | none | **M8 knob:** instruction noise (typos, ambiguity) and tool noise (transient errors) on toy worlds, with the clean-vs-noisy gap as the promotion criterion |
| Fork for side-effect-free judging; pause and snapshot (Kimi) | WorldRef snapshot/restore; C3 branches restore; verifier on the end state | same pattern; consider verifying on a forked snapshot so judging never mutates the live world |
| Critical-path latency for multi-agent work (DeepSeek DAG, Kimi critical steps) | P28 multi-agent World; efficiency term is generic | **M6/M13:** an efficiency metric = critical-path steps (main agent + slowest peer per stage), not total steps |
| Context management as an RL action; compaction segments (CompactionRL, MiniMax) | The harness is recorded per estimate; no compaction events | **M3/M10:** a `context_compaction` trace event so steps.jsonl credit can span segments (cross-trajectory GAE) |
| Vary harness and context mechanisms per task group (Kimi) | `harness@version` on every credit estimate; one harness per GRPO group | consistent; M12 should sample harnesses across groups |
| Environment building as an RL-able skill (StepFun); strong models mass-produce Docker environments (Ling) | M8 Environment builder role (scripted for toy) | **M8:** the builder role through Harbor installed agents; its success rate as a forge metric |
| Group-size-1 environments need a critic (SAO) | Our credit assumes GRPO groups | **M12 note:** single-feedback enterprise workflows (one real case, one outcome) are exactly SAO's setting. Φ and C3 values are candidate baselines |

## 6. Prioritized actions (cheap to expensive)
1. **M6:** `truncated` failure kind → mask or Φ-bootstrap (small, credit-only).
2. **M1/M6:** attribution of environment crashes the agent caused (`agent_broke_env`), never masked.
3. **M8 Adversary:** soft-hack negatives; **C4:** a minimality/scope rubric item.
4. **M6:** a GAR-style advantage redistribution option (mass-conserving, hack-zeroing).
5. **M10/M11:** a rollout record schema with token ids, behaviour logprobs and policy version per span.
6. **M13:** a critical-path efficiency metric; a pass-rate-jump hack alarm.
7. **M8:** a noise-injection generator knob; environment-builder success rate.
8. **task04 re-versioning:** perf inputs varied per round.
