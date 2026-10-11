# M12 train: what the training libraries do better, ranked fixes (2026-10-11)

Sources: `RL-Environments/06-frameworks-and-env-code/infra-*`:
- prime-rl, verl, NeMo-RL, SkyRL, rLLM, AReaL, ROLL, slime, OpenRLHF and TRL are Apache-2.0;
- checkpoint-engine is MIT.

## Why the first 0.6B run barely learned (`reports/labstack/m12/m12_gpu_gate.json`)

- **15 of 20 steps had no signal:** `zero_variance_groups = 1.0` and `train/loss = 0.0`.
- **Signal on 5 steps only,** each with reward ≤ 0.047.
- **Entropy 0.07–0.11 nats/token,** so the rollouts in a group were near-duplicates.
- **lr 1e-6 × 20 Adam steps** is a cumulative drift of at most ~2e-5.
- **No eval and no grad-norm log.**

The async machinery was healthy: staleness ≤ 1, `rollout_probs_diff` about 0.005, 0 masked tokens. **The problem was
the signal and the optimization budget.**

**Fixed on 2026-10-11:**
- The gate gained `--task wordfile` (a learnable seeded task) and held-out eval before and after training (`--eval`).
- The trainer logs `train/grad_norm` (pre-clip) and `train/lr`.

## Ranked improvements

| # | Change | Source | I / E | Mode |
|---|---|---|---|---|
| 1 | Difficulty-aware task sampling (`train/curriculum.py` as the scheduler's `sources`): per-prompt history; all-0 / all-1 prompts retired to side pools; mostly sample 0 < p < 1 | prime-rl `orchestrator/curriculum/samplers/pool.py:18`, `gates/adv.py:15` | H / M | PATTERN |
| 2 | Dynamic sampling that cannot deadlock: `filter_zero_variance` plus a `keep_when_insufficient` fallback after N blocked barriers, and filtered-reward counts | slime `rollout/filter_hub/dynamic_sampling_filters.py:9,18`, `sglang_rollout.py:498-517`; verl `algorithm.py:59`; OpenRLHF `train_ppo_ray.py:601-604` | H / S | PATTERN |
| 3 | Learning-signal metrics: grad norm (**done**), mean \|Â\|, share of tokens with nonzero advantage, post-update KL(π_θ ‖ π_rollout), lr (**done**) | prime-rl `trainer/rl/loss.py:138,180-186` | H / S | PATTERN |
| 4 | Periodic eval inside `Runtime` (`eval_every`, held-out seeds, pass@k on the same engine version) | verl `ray_trainer.py:1438,1727`; prime-rl `orchestrator/eval_source.py:24,53,99` | H / M | PATTERN |
| 5 | LR schedule, warmup and optimizer knobs; refuse lr > 1e-4 for HF models (the default 3e-3 is a TinyPolicy value) | prime-rl `trainer/scheduler.py:17-35` | M / S | PATTERN |
| 6 | Temperature consistency: the trainer recomputes logprobs with `cfg.temperature` and ignores `Turn.sampling["temperature"]`; use it or refuse a mismatch | Crucible-specific | M / S | — |
| 7 | Entropy: true trainer-side entropy, an optional bonus, clip-cov / kl-cov, a forking-token mask | verl `core_algos.py:1743,1848,2108`; TRL `grpo_trainer.py:1341,2978,3072` | M / M | PORT |
| 8 | Sparse-reward estimators: `passk` advantages, batch-std scaling, REINFORCE++ global whitening | verl `core_algos.py:471,536,694`; TRL `grpo_trainer.py:2696-2703`; slime `ppo_utils.py:620` | M / S | PORT |
| 9 | GiGPO step-level credit (anchor states across rollouts of a task) | ROLL `pipeline/agentic/utils.py:61,183-210`; rLLM `unified_trainer.py:345` | M / M | PORT |
| 10 | Wire the DAPO overlong penalty (`objectives.overlong_penalty` is defined but never called) | verl `reward_manager/dapo.py:45,108`; TRL `:2416`; NeMo-RL `grpo.py:369` | M / S | PATTERN |
| 11 | LoRA RL (adapter-only publishes, about 10× the lr; vLLM LoRA hot-load) | prime-rl `trainer/lora.py:97,193,210` | M / M | PATTERN |
| 12 | General varlen sequence packing (the run used 30–37k forward tokens per step to train 3–6k) | NeMo-RL `data/packing/algorithms.py:179,304`; prime-rl `rl/data.py:180` | M / M | PATTERN |
| 13 | Weight-sync speed: CUDA IPC buckets when colocated, NCCL broadcast when disaggregated, RDMA P2P at scale | slime `update_weight_from_tensor.py:45-52`; NeMo-RL `weight_sync/factory.py:48`; prime-rl `transports/weights/nccl.py`; checkpoint-engine `ps.py:176,237` | M / M–L | PATTERN |
| 14 | An unbiased KL gradient (`k3+` / `k2`; trainer.py:397 uses k3 directly) | verl `core_algos.py:2188-2214` | L / S | PORT |
| 15 | IPO absolute-probability trust region; sequence mask/truncate modes | prime-rl `loss.py:145-190`; TRL `:2589-2635` | L / S | PORT |
| 16 | Retry with backoff in `VLLMServerEngine.generate` before masking | AReaL `remote_inf_engine.py:1105-1115` | L / S | PATTERN |

**Already at parity:**
- partial rollouts and interruptible generation (AReaL, slime);
- the mismatch corrections TIS / IcePop / MIS / CISPO / GSPO;
- token vs sequence normalization (Dr.GRPO);
- bucketed broadcast sync, offload, DP balancing, checkpoints, sinks.

## The five changes most likely to make a small model learn

1. **A task the model can sometimes solve** (measure the pass rate first; the per-prompt rate in 0.1–0.7), plus
   per-prompt pools.
2. **n = 16, B = 8–16, `filter_zero_variance` with the fallback.** At p = 0.05, a group has variance about 19% of the
   time at n = 4 and about 56% at n = 16.
3. **On-policy distillation is already built:** `TeacherPool` / `VLLMTeacher`, `opd_coef ≈ 1`, with a Qwen3-4B/8B
   teacher of the same vocabulary. It gives every token a dense signal even when all rewards are 0.
4. **200+ steps at lr 2–5e-6 with warmup, or LoRA r = 32 at 2e-5–5e-5,** with grad norm / \|Â\| / KL logged.
5. **Denser reward and wider exploration:** a positive format term, per-ticket progress (`turn_weight > 0`) or GiGPO,
   and rollout temperature 1.1–1.2 matched on the trainer.
