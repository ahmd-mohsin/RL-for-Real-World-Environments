# M12 infra: every option of "Mainstream Technology Routes in Agentic RL Infrastructure" (2026-10-10) against the code

Source: the article (11 labs' 2025–2026 reports), the earlier deep read `docs/research/m11-m12/06-article-systems.md`
(52-item checklist, built in M12), and the code in `src/crucible/train/` as of 2026-10-12. Paths are relative to
`src/crucible/`. **Have** = implemented and tested; **Part** = implemented without the article's full form or untested at
scale; **Gap** = absent. The gaps are built in M12-infra batches A–D (bottom).

## Sync or async: both, as orthogonal axes

The article's key structural point (§5 route ②): placement (colocated / disaggregated) and synchronization (sync /
async) are two axes. Crucible keeps them orthogonal: `RuntimeConfig.topology` sets where training and generation
meet, and the scheduler's staleness knobs set how far behind data may be. One scheduler object
(`train/scheduler.py`) makes every decision; the asyncio runtime (`train/runtime.py`) and the deterministic
discrete-event simulator (`train/sim.py`) both drive it, so every mode is testable without GPUs and sweepable before
GPU time.

| Mode (article name) | Crucible setting | Status |
|---|---|---|
| Synchronous on-policy | `topology=colocated_sync` (staleness 0, spans 1) | Have |
| One-step off-policy | `disaggregated_stream`, `max_staleness=1`, `admission_staleness=1` | Have |
| Colocated partial rollout, family A (Kimi K3, MiMo, DeepSeek-V4.1, Ring, Ling) | `colocated_partial` + `partial_barrier ∈ {groups, completion_frac λ, token_budget Φ}`, `max_cuts` σ, `resume_keeps_kv` | Have (batch A) |
| Disaggregated fully streaming, family B (GLM-5, DORA, StepFun, Qwen3.5) | `disaggregated_stream`, `publish_every=K`, `reset_optimizer` on publish | Have |
| Windowed FIFO, family C (MiniMax) | `buffer_policy=windowed_fifo`, `fifo_window_frac=0.3` | Have |
| Elastic hybrid (LongCat DORA: dedicated generation group + elastic group switching roles each step) | `topology=elastic`, `elastic_engines` | Have (batch A) |

## Full option matrix

### Planes and control (art. §1)
| Option | Where | Status |
|---|---|---|
| Control plane: admission, selection, staleness, mix | `train/scheduler.py` | Have |
| Data plane: tags vs values pool, status machine | `train/pool.py` | Have |
| State plane: run checkpoint at the barrier; mid-episode World checkpoints | `train/state.py`, `runtime/checkpoint.py` | Have |
| Weight plane: in-process, checkpoint dir, NCCL/gloo bucketed double-buffered broadcast, atomic commit, versions, elastic join, sharded owners | `train/weights.py`, `train/weightsync.py` | Have (validated gloo multi-process; NCCL on the lab only) |
| Engines: TITO, version stamp, abort, pause, sticky routing, health, eviction, remote HTTP engines; vLLM / SGLang HTTP engines; HF engine | `train/engine.py`, `gateway.py`, `vllm_http.py`, `sglang_http.py`, `hf_engine.py` | Have |
| Env off the GPU pool: env worker threads, remote env servers (NeMo-Gym protocol, OpenEnv WebSocket), rollout workers with leases | `train/envpool.py`, `train/workers.py`, `world/remote.py` | Have |
| Launcher: processes, supervision, restarts (Ray-free) | `train/launch.py` | Have. **Gap**: Ray launcher, multi-node placement planner |
| Monitors: staleness, mismatch, length-bias KS, entropy, zero-variance share, hack rate, bubbles, pass-rate alarm, circuit breaker, sinks | `train/monitors.py`, `train/sinks.py`, `runtime.py` | Have |

### Placement and offload (art. §2.1, §3.1)
| Option | Where | Status |
|---|---|---|
| Colocated: trainer state offload `cpu` / `nvme`; KV `discard` (sleep) / `cpu_pool` | `RuntimeConfig.train_state_on_rollout`, `kv_on_train`; `TorchTrainer.offload` | Have |
| Memory-split single GPU (engine fraction + resident trainer) | runtime placement note; lab runner | Part (lab) |
| Disaggregated pools with independent scaling | gateway + launcher + weightsync | Part (no planner for N nodes × G GPUs) |
| Elastic co-card (DORA) | `topology=elastic` | Have (batch A) |
| Trainer at scale: reference torch, DDP-style DP, bridges to verl / TRL | `train/trainer.py`, `bridges/` | Part. **Gap**: native FSDP2 backend (sharded params/grads/optimizer) |

### Long tail, granularity, length bias (art. §3.2)
| Option | Where | Status |
|---|---|---|
| Dispatch unit = one rollout; groups are admission and advantage units only | scheduler | Have |
| Per-source in-flight caps (DeepSeek sample-level scheduling), source mix by deficit | `inflight_cap`, `mix` | Have |
| Early-short drop, per-source caps, KS length-bias monitor | `drop_early_short`, monitors | Have |
| Partial-rollout boundary: B groups selectable | `colocated_partial` | Have |
| Boundary by completion fraction λ (Kimi K3) | `partial_barrier=completion_frac` | Have (A) |
| Boundary by global token budget Φ, cross-version buffer (Ring C3PO++) | `partial_barrier=token_budget` | Have (A) |
| Retention σ: evict rollouts cut more than σ times (Ring) | `max_cuts` (runtime + sim) | Have (A) |
| Resume at a token boundary without re-prefill (DeepSeek-V4.1) | engine capability `resume_keeps_kv` + runtime/sim accounting | Have (A; real KV reuse needs an engine that keeps it) |
| Token-budget micro batches, DP length balancing | `train/batching.py` | Have |
| Prefix sharing (MiniMax prefix-tree merging) | `TrainerConfig.prefix_sharing`, `shared_prefix_logprobs` | Have |

### Staleness (art. §3.3)
| Option | Where | Status |
|---|---|---|
| Version cap on the oldest version (GLM τ; worked example) | `Tag.staleness`, `staleness_strategy=drop` | Have |
| Dropless wait (MiMo) | `staleness_strategy=wait` (open-group staleness, bound groups first, `bound_overflow`) | Have (fixed in A: B.21) |
| Pool gating S×B×n untrained (Ling staleness manager) | `admission_staleness` | Have |
| Per-source τ | `max_staleness_by_source` | Have |
| Max off-policy share per batch (DeepSeek) | `max_offpolicy_frac` | Have |
| Algorithm-layer masks (all of §3.3.3.3) | `train/mismatch.py` | Have |

### Train–inference mismatch (art. §3.4)
| Option | Where | Status |
|---|---|---|
| TIS, IcePop, KPop, DIS, MIS-PO, HisPO, CISPO, GSPO, PPO (+ dual clip) | `mismatch.py` | Have |
| MiMo four-bound mask with entropy-adaptive bounds | `mimo4` + `TrainerConfig.entropy_adaptive` | Have (B) |
| CISPO M2 asymmetric `[0, 1+ε]` | `cispo` param | Have |
| FP32 LM head (BF16 in, FP32 out) | `train/hf.py lm_head_fp32` | Have |
| QDQ to the rollout grid after each update (MiMo MXFP4; Kimi QAT) | `weights.QDQ` (mxfp4 / fp8; trainer weights too) | Have (B) |
| Deterministic / batch-invariant kernels (DeepSeek, GLM top-k) | `torch` determinism in tests; engine flags not wired | Part (engine-side; recorded, not enforced) |
| R3 routing replay; top-p candidate replay; routing confidence Σ_k | `routing_replay=r3`, `Turn.candidates`, `moe/routing_confidence` | Have |
| Concatenated routing replay across version spans (DeepSeek) | R3 per call | Part (spans carry per-token routing; no cross-checkpoint test) |
| TITO end to end (no detok/retok) | `train/chat.py` codecs, engines | Have |

### Algorithms (art. §3.5)
| Option | Where | Status |
|---|---|---|
| GRPO (mean / std), Dr.GRPO, RLOO | `objectives.group_advantages` | Have |
| ReMax, REINFORCE++ (global-batch normalized) estimators (Hunyuan one-click set) | `estimator=remax|reinforce_pp` | Have (B) |
| DAPO: clip-higher, dynamic sampling, token-level loss, overlong filter + soft penalty | `eps_h`, `filter_zero_variance`, `token_mean_global`, `overlong_penalty`, truncation masking | Have |
| Loss aggregation: token_mean_global, seq_mean_token_mean, prompt_mean, seq_ratio, const_max_len | `objectives.aggregate` | Have |
| Critic mode: single rollout (n=1), token GAE, skip-observation, cross-segment (CompactionRL), truncation bootstrap, length-adaptive λ, TTUR, frozen attention | `objectives.gae`, `train/critic.py` | Have |
| Value pretraining (Seed, SAO "scaling value pretraining") | `critic.pretrain_value` | Have (B) |
| KL k3 to a reference; per-source KL | `kl_coef`, `kl_coef_by_source` | Have |
| On-policy distillation: token MOPD, full-vocabulary OPD, teacher routing, vocab gate | `train/teachers.py`, `opd_*` | Have |
| Reward-to-go with per-step rewards and a trajectory baseline (MiniMax M2: (s_t, a_t) atomic units) | `credit_unit=reward_to_go`, `Turn.step_reward` | Have (B) |

### Rewards and credit (art. §3.6)
| Option | Where | Status |
|---|---|---|
| Outcome + tool-format penalty + incomplete-trajectory penalty | `tool_format_penalty`, limit penalty | Have |
| Speed reward h(T/T_baseline) (MiniMax M2.5) | `speed_coef` (gated) | Have (B) |
| Group grading hook (GAR, tournaments, hack zeroing) | `RuntimeConfig.group_fn` + `train/group_ops.py` (gar, grs, bt_pairwise, tournament, zero_hacks, compose) | Have (B) |
| Rubric / judge scoring channel with deadline | `aux_fn` | Have |
| DAG critical path latency (DeepSeek Agent Team) / Critical Steps (Kimi PARL) | `credit.critical_steps`, `credit.dag_critical_path`, `latency_coef` | Have (B) |
| Frozen sub-agents (PARL) | `TrainerConfig.frozen_actors` (reference trainer; was bridges-only) | Have (fixed in B) |
| Turn credit from M6 ΔΦ | `credit.py turn_weight` | Have |

### Reward hacking and environment noise (art. §3.7, §3.8)
| Option | Where | Status |
|---|---|---|
| Online interception returning dummy content (GLM-5.2) | `world/intercept.py` | Have |
| Full-tier audit sample; pass-rate jump alarm | `audit_rate`, `PassRateAlarm` | Have |
| Sandbox hardening, firewall, custody, separate verifier | M1 items 1, 5; guard | Have |
| Infra failure = masked; agent-broken env = 0 | M10 failure taxonomy | Have |
| Flaky-task filter: reference reruns K times must agree (MiMo 8×) | M9 flake checks on oracles | Part → **Gap** (D: task admission filter in the training source) |
| Noise-robust curriculum: instruction and tool noise with rising intensity (LongCat) | — | **Gap** (D) |
| Pause / fork / snapshot of sandboxes (Kimi AgentENV) | World snapshot/restore, S6 pause | Have |
| Context management as an RL action; compaction segments | `ContextConfig(mode=action)`, segments | Have |

## Gap build plan (M12-infra)

- **A. Scheduling and topology**
  - `elastic` topology (DORA): a resident generation group plus an elastic group that switches between generation and training each step.
  - Partial-rollout boundary policies `barrier ∈ {groups, completion_frac λ, token_budget Φ}`, with retention σ (evict rollouts cut more than σ times).
  - A KV-preserving resume path (`Engine.capabilities()["resume_keeps_kv"]`; no re-prefill accounted).
  - All of these in both the runtime and the simulator, with sim sweeps.
- **B. Algorithms and credit**
  - ReMax and REINFORCE++ estimators.
  - Reward-to-go with per-step rewards.
  - Entropy-adaptive mimo4 bounds.
  - Speed reward.
  - Critical-steps / DAG critical-path latency.
  - Concrete group functions: GAR, GRS, BT-pairwise, tournament, hack zeroing.
  - FP8 / MXFP4 QDQ transforms.
  - Value pretraining.
- **C. Scale-out**
  - Native FSDP2 trainer backend.
  - Ray launcher (optional dependency).
  - A placement planner (`ClusterSpec`: nodes × GPUs → engine and trainer groups per topology).
  - A multi-process, multi-"node" gloo run of the full loop.
  - Large-scale simulator sweeps: 1k+ engines and 32k concurrent environments (DORA's scale), measuring bubbles and staleness.
- **D. Environment noise**
  - A flaky-task admission filter (K reference reruns must agree).
  - A noise-injection curriculum wrapper over World tools and instructions (LongCat), raised by the clean/noisy gap.
