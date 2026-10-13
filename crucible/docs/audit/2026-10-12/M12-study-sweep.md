# M12 study: every local source, read for training, and the picks

Five sweeps read every local source for M12 `train` at code level, with probes that run the frameworks' own code (loaded
from the pinned checkouts with stubs) against Crucible's on identical tensors. Four defect hunts audited the module
itself (≈9.7k lines). Each is its own document:

| Group | Sources | Document |
|---|---|---|
| G1 | the qingkeai 2026 agentic-RL-infra article (Chinese original; all 56 display formulas re-extracted from the HTML), the Zhihu framework comparison (digest `07-articles/zhihu-rl-frameworks-comparison-digest.md`), every other article for training content | `M12-sweep-G1.md` |
| G2 | Xiaomi's verl fork (colocate_async, R3, mimo4), the MiMo v2.6 RL release, verl (core_algos, agent loop, fully-async, rollout correction) | `M12-sweep-G2.md` |
| G3 | slime, AReaL, ROLL (staleness manager, partial rollout, decoupled PPO, weight update, router) | `M12-sweep-G3.md` |
| G4 | prime-rl, NeMo-RL, SkyRL, rLLM, verifiers → prime-rl | `M12-sweep-G4.md` |
| G5 | TRL, OpenRLHF, checkpoint-engine, production 8-GPU recipes, multi-GPU readiness and the REPACSS plan | `M12-sweep-G5.md` |
| hunts | the learning math (H1); control and data planes (H2); rollout plane, token fidelity, weight plane (H3); scale-out and bridges (H4) | `M12-study-defects-H1.md` … `-H4.md` |

## What held

- **Per-token loss math matches every upstream.** Max |Δgrad| ≤ 1e-8 between Crucible and verl, MiMo-verl, NeMo-RL,
  SkyRL, prime-rl, TRL and AReaL for PPO (+dual clip), TIS, IcePop, CISPO, GSPO, DIS, mimo4, MIS-PO and AReaL's
  decoupled three-policy loss; the article's worked numbers (KPop 4.05e-5 / 0.4534, MIS-PO 0.67 / 0.018 / 2e-9, DAPO
  overlong, CompactionRL (γλ)^{N>s}, SAO skip-observation GAE, CISPO's stop-gradient) reproduce.
- **Global token normalisation is exact** across micro batches and DP ranks (verl's `global_batch_info`), and better
  than SkyRL for the KL term.
- **Scheduling core**: 20,000 fuzz seeds on supported configurations: nothing trained twice, no masked member trained,
  staleness never above τ under `wait` / `drop`; Crucible's τ is exact where AReaL's staleness manager (a capacity cap)
  trains 6–35 versions stale on the same workload.
- **Exact TITO** on the real Qwen3 / Qwen3-Instruct / Qwen2.5 templates; KV-cached resume equals a full forward.
- **Sharded training**: FSDP / HSDP parity at 2–8 gloo ranks within 1.4e-6, DCP resharding 8→4→3 exact, tied
  embeddings correct.

## What every group found (the most certain findings)

- **One extreme behaviour logprob NaNs the whole model** (G2, G3 D7, G4 #1, H1 V2): `torch.where(gate, r·Â·logp, 0)`
  back-propagates 0·inf; nothing checked a non-finite gradient.
- **Aggregations other than the token mean** ran the whole batch as one forward, were refused under DP, and counted
  calls / segments as sequences (G2 H4/H8, G4, G5 D11, H1 V4/V13); GSPO and HisPO were only correct under a
  non-default aggregation (G1 #3).
- **The trainer never checked token completeness or the sampling temperature** (G2 H9, G4 #7, H1 V3, H3 V21).
- **GAR after GRS / BT / tournament** treated passing solutions as failures; two GAR implementations disagreed (G1 #8,
  G2 H1, H1 V1).
- **`max_offpolicy_frac` and `windowed_fifo`** broke τ or deadlocked; their tests were vacuous (G1 #2/#6, G3 D1–D4,
  H2 V4–V6, V11).
- **A data-parallel rank with no rows hangs every trainer** (G5 D1, H4 V1); a parameter one rank never reaches hangs
  DDP and mis-updates FSDP step mode (G4 #2, H4 V2).
- **The vLLM top-p nucleus was rebuilt wrongly** (G2 H3, H1 V10, H3 V3), and the **framework bridges (verl, TRL,
  SkyRL, rLLM) cannot run** against the current upstreams (G2 H7, G4 #4/#5, G5 D4, H4 V19–V28).
- **Multi-node has no launch path and the 30B-A3B load does not fit** (G5 D2/D5, H4 V4/V9); REPACSS is 4 × H100 NVL
  per node, so 8 GPUs means two nodes.

## The picks

DEPEND = use the framework, PORT = copy code (licence in the group document), PATTERN = rebuild the idea. "Built" = built
in this audit round.

| # | Pick | Source | Kind, effort | Status |
|---|---|---|---|---|
| 1 | Clamp log ratios at ±20 before exp, finite zero weights at gates, drop non-finite behaviour tokens, skip a step on a non-finite grad norm | SkyRL `torch_utils.py:188`, AReaL `fsdp_engine.py:742`, slime `model.py:668` | PORT, S | built |
| 2 | Every aggregation as an exact per-rollout row weight (micro-batches, DP); `auto` picks GSPO / HisPO's own unit; HisPO's per-group G·max\|y\| | verl `global_batch_info`, SkyRL `ppo_utils.py:1152`, NeMo-RL `clipped_pg.py:169`, MiMo `actor.py:218` | PATTERN, M | built |
| 3 | Sequence statistics per rollout (`seq_ids`): GSPO ratio, MIS-PO / HisPO trajectory gates | verl / MiMo one row per trajectory | PATTERN, M | built |
| 4 | Trainer-side validity gate and temperature check | prime-rl, multi-harness article | PATTERN, S | built |
| 5 | Reverse-KL gradient for the k3 term, clamps | NeMo-RL `loss_functions.py:658` | PORT, S | built |
| 6 | One GAR; pass set from the raw outcome; stage score as quality | MiMo GAR (article §3.6.3.2c) | PATTERN, S | built |
| 7 | Truncation bootstrap replaces R with V(s_T); critic on shaped and step rewards | StepFun (article F52), AReaL `test_ppo_actor_truncation.py:282` | PORT, S | built |
| 8 | QDQ with fp32 master weights; MXFP4 ties-to-even; HF parameter names | Megatron master weights, OCP MX | PATTERN, M | built |
| 9 | Missing objectives: prime-rl `ipo`, sequence-level TIS / mask, CISPO unified mask, DAPO overlong shaping + filter, Bessel std, REINFORCE++ baseline | prime-rl `loss.py`, TRL, NeMo-RL, MiniMax-M1, verl `dapo.py`, OpenRLHF | PORT, S | built |
| 10 | vLLM nucleus from processed logprobs (finite entries); raw mode refused | vLLM `--logprobs-mode` | PATTERN, S | built |
| 11 | DP: metric keys agreed, rank-local parameters zero-filled, group-relative source rank, critic / teacher replicated | own | S | built |
| 12 | Padded micro-batch budget | verl / NeMo-RL | PATTERN, S | built |
| 13 | Scheduler: off-policy cap after the at-bound rule, window over untrained groups, bounded refills and zero-variance resampling, contained rollout failures | own (G3 D1–D5, H2) | S | see `M12-train.md` |
| 14 | TITO keyed per conversation; vLLM drain before a weight load; gateway weight snapshots; SGLang R3 on slime's row contract | slime | PORT, M | see `M12-train.md` |
| 15 | Bridges against current verl / TRL / SkyRL / rLLM | upstream dataclasses | DEPEND, M | see `M12-train.md` |
| 16 | Multi-node launch, rank-0 load + broadcast, FSDP offload, node-local meshes, checkpoint GC, REPACSS recipe | torch DCP, verl / NeMo-RL Slurm recipes | PATTERN, M | see `M12-train.md` |
| 17 | Bucketed broadcast + CUDA IPC into vLLM | checkpoint-engine (MIT) | DEPEND, M | queued |
| 18 | Varlen packing, OpenAI proxy for black-box agents, Ling focus / StepFun asymmetric mapping / PARL annealing | verl, AReaL `experimental/openai`, article §3.6 | PATTERN, M each | queued |
