# 06: The systems side of "Agentic RL Infra" (Wei Ziyang, 2026), deep read for M12

- **Source:** https://qingkeai.online/blog/2026-Agentic-rl-infra (mirror of https://weiziyang.wiki/posts/post-training-rl/agentic-rl-infra-technical-roadmap-overview), in Chinese. Author: Wei Ziyang (韦子扬).
- **Local copies:**
  - `RL-Environments/07-articles/qingkeai-2026-agentic-rl-infra.md`: text with the formulas stripped.
  - `RL-Environments/07-articles/raw/qingkeai-2026-agentic-rl-infra.html`: the 56 display formulas, recovered from `<div class="language-math">`.
- **Also read for this note:**
  - all 12 figures of the article. Fig. 1: four planes. Fig. 2: colocated vs disaggregated. Fig. 4: Kimi agentic framework (inferred, see §1.2). Fig. 5: Ant ASystem. Fig. 6: MiMo-V2.6 RL infra. Fig. 7: MiniMax Forge. Fig. 8: LongCat DORA. Fig. 9: windowed FIFO. Fig. 12: MiMo GRS/GAR.
  - the interactive windowed-FIFO simulator (React TSX) embedded in the original page.
  - the MiniMax Forge blog (footnote 15).
  - the LongCat-Flash-Thinking blog (footnote 10).
  - the code the article cites as MiMo's open reproduction (footnote 13, `XiaomiMiMo/verl`): `verl/trainer/ppo/v1/trainer_colocate_async.py`, `trainer_base.py` and `replay_buffer.py`.
- **Companion:** `../2026-10-09-agentic-rl-infra-survey.md` is the summary-level read. This note goes deeper on topology, async control flow, granularity, weight sync, engines, sandboxes, mismatch formulas, middleware and compaction. Each item ends with what Crucible M12 should implement.
- **Section references:**
  - "art. §x.y" is the article's own numbering.
  - "art. Table" is the §4 cross-lab table. Several system details (token-level WAL, Prefix Tree Merging 40×, "new requests only go to the old version (load 63%)", PD disaggregation + CPU KV swap) appear **only** in that table and are not explained in the prose.
- **Read:** 2026-10-09, in full, including formulas and figures. No Crucible code was changed.

---

## 0. The frame: four planes (art. §1, Fig. 1)

The article splits an agentic RL system into four planes. Fig. 1 makes the components explicit:

| Plane | Article's definition | Fig. 1 components | Crucible M12 owner |
|---|---|---|---|
| **Control** | global scheduling, resource allocation, version management, failure recovery, observability | Scheduler (task scheduling, prioritization), Version Manager (model / data / code), Resource Allocator (GPU / CPU / storage), Failure Recovery, Observability (metrics, logs, traces, alerting) | `train/scheduler.py` (dispatch, admission, staleness), `train/versions.py`, monitors |
| **Data** | trajectories, logprobs and rewards moving between rollout, queue, verifier and trainer | Rollout GPU pool → *Sample Queue / Replay Buffer* → Reward & Verifier (rule-based verifier, GRM/judge, filtering/anti-hack) → Trainer | `crucible.rollout/v1` (exists, `train/rollout.py`), `train/buffer.py` |
| **State** | KV cache, environment state, tool sessions, partial rollouts | KV cache (prefill/decode), the environment and KV state between rollout pool and environment pool | World checkpoint (`runtime/checkpoint.py`, exists), policy `state_dict()`, paused-call registry |
| **Weight** | new weights from trainer to rollout engine: sync, conversion, version switch | Checkpoint / Weight Store (versioned checkpoints, S3/NFS) → "weight sync / hot swap" → rollout pool | `train/weights.py` (WeightPublisher) |

Fig. 1 lists five core challenges: **staleness, train–inference mismatch, long-tail rollout, weight sync, observability**.

**M12:** organize the package along these four planes. Each plane gets one interface and one owner, and every interface carries the **policy version** as a first-class field. The version is the join key across all four planes.

---

## 1. Topologies (art. §2.1, §3.1, Table)

### 1.1 Definitions and the article's key structural claim

- **Colocated (共卡):** rollout and training time-share the same GPUs.
  - *Pain:* memory contention. KV cache vs params, grads and optimizer state, worst at 1M context; KV usually has to be discarded to make room for training. Phases are coupled in time, the throughput ceiling is low and scaling is limited.
  - *Gain:* fewer GPUs, no weight transfer across machines.
- **Disaggregated (分离):** separate rollout and training GPU pools joined by distributed trajectory storage (Data Pool / SampleQueue).
  - *Pain:* cross-cluster weight-sync latency, a pool-split ratio that is hard to tune, more resources.
  - *Gain:* each pool scales elastically and uses its own best parallelism.
  - The article's structural claim (§3.1.2): once the pools are split, the buffer between them *must* run as an asynchronous producer–consumer, or the two pools wait on each other. **Asynchrony is an intrinsic property of disaggregation, not an optional add-on.**
- **Elastic hybrid (弹性共卡):** the system stays disaggregated overall, but part of the GPUs time-share.
  - LongCat DORA splits the cluster into a **dedicated generation group** (always inferring) and an **elastic role group** that switches between generation and training roles *within each step*, via in-process context switch plus offload.
  - The article calls this "disaggregated + partial colocation". The first DORA blog called it "asynchronous elastic colocation".
- **Orthogonality** (art. §5 route ②): colocated/disaggregated and sync/async are independent axes.
  - GLM and DORA are disaggregated + async.
  - DeepSeek-V4.1 and Ling 2.6 are colocated + async.
- **Cross-camp consensus** (§3.1.4): **environment and sandbox execution leave the GPU pool** and scale elastically on CPU (DeepSeek DSec, Ling's thousands of ASandbox replicas, LongCat's CPU-idleness-aware RPC). Environment latency is set by external systems and must not hold GPUs idle.

### 1.2 Every topology, per lab

| Lab | Placement | GPU count / concurrency | Memory juggling and specifics | Stack |
|---|---|---|---|---|
| **Kimi K3** (§3.1.3) | colocated | keeps each 1M-context experiment within **"a few hundred GPUs"**; "resource efficiency under limited compute is the first goal" | After a **training round**, weights and optimizer state are **offloaded to NVMe**, freeing CPU DRAM for an **external KV-cache pool**. After a **rollout round** that pool is released. **Reference and other forward-only non-policy models live on CPU** and, when needed, are materialized by **borrowing the policy model's FP32 gradient buffer** (the buffer is idle during forward-only passes). | inherits K2 infra |
| **Kimi K2.5** (Fig. 4, inferred) | — | — | Fig. 4 sits between the Kimi and Hunyuan paragraphs and matches K2.5's description. A **Rollout Manager** drives N "single agent tasks". Each task has pluggable components (Toolset, Judge, Prompt & Instruction Enhancement), a Core Agent Loop, and an Env Pool holding **black-box envs behind an LLM Gateway** and **white-box envs**. The task talks **token-in / token-out** to an **Inference Engine Service** and feeds a **Training Engine Service** with a "Mismatch Correction" edge. A "Recursive Call" edge (subagents) loops back into the task. The attribution to K2.5 is my inference: the figure caption only says "Figure 10: Overview of our agentic RL framework". | — |
| **Hunyuan Hy3** (open recipe) | colocated | **128 × H20**. Rollout needs **TP = 16** to fit a single instance. | While training is idle, params, optimizer state and grads are all **fully offloaded to CPU** so rollout gets the memory. UniRL supports both colocated and split cards (Table). | Megatron-LM + vLLM (verl + Megatron-Bridge) |
| **DeepSeek-V4.1** | colocated, time-shared | DSec manages hundreds of thousands (V4) to **millions (V4.1)** of concurrent sandboxes | Agent sandboxes and worker containers run on **DSec, outside the preemptible GPU training pool**: colocated + environment off-GPU. | in-house |
| **Ant ASystem** (Ling/Ring) (Fig. 5) | colocated "to the extreme" | thousands of ASandbox replicas | The **Hybrid Runtime** co-locates the training and inference engines in every GPU worker. **AMem** is a unified memory library for RL. **AState** does zero-redundancy P2P weight exchange, **1T parameters within 10 s**, sub-second updates. **ASandbox** is a serverless on-demand sandbox engine (HTTP and MCP protocols; math and logic verifiers, code sandbox). **AReaL** is the algorithm layer (single controller + SPMD). | Megatron + SGLang |
| **MiMo-V2.6** (Fig. 6) | not stated; the article infers **colocated + async** | 7k+ environments (Table) | Interrupts in-flight sequences once a training batch is full and re-prefills next phase, which fits time-sharing. The open recipe's default is `colocate_async`: "Trainer and rollout are colocated". After sampling it **aborts requests, puts replicas to sleep, discards weights and KV**; after training it updates weights and resumes (control flow in §2.3). **Fig. 6 components:** a "Payload Porter" containing a control plane (**Sample Mixer**, **Packer**) and a data plane (**Distributed Trajectory Storage**); a **Harness Pool** (Harness A/B/C instances, endpoints, agent loops) sending "Tokens & Δimages"; an **Inference Engine** with Prefill → Decode (DFlash), **Context Cache**, **L2Cache** and capture of **MoE IDs / Top-p**; a **Training Engine** with **SWA CP** (context parallelism for sliding-window attention) and **Loss Fusion**, emitting **QDQ weights** back to inference. | Megatron-LM + SGLang |
| **GLM-5** (slime) | **disaggregated** ("decouple training and inference engines onto different GPU devices") | **1k+ concurrent rollouts**; 10k+ SWE environments | A central **Multi-Task Rollout Orchestrator**; each task implements its rollout and reward logic as an **independent microservice** registered with it. slime = **server-based rollout + heartbeat fault tolerance + PD (prefill/decode) disaggregation** + router-level service lifecycle + a **TITO gateway** (token-level train/inference correspondence). | slime (Megatron + SGLang) |
| **MiniMax Forge** (Fig. 7) | disaggregated | hundreds of scaffold types, thousands of tool-call formats, **100k+ scaffolds/environments**; **millions of samples per day at up to 200k context** (Forge blog) | **Middleware** = Gateway Server + Data Pool (§6.1). Engine side (Forge blog): **heterogeneous PD disaggregation** (to stop MoE scheduling interference); a **global L3 KV-cache pool on a DFS** with a **cost-aware scheduler** weighing queueing delay against cache migration cost; **MTP speculative decoding** whose MTP heads are fine-tuned continuously against the moving policy with a Top-K KL loss; **Prefix Tree Merging, a 40× training speedup** (§3.4). | in-house |
| **LongCat DORA** (Fig. 8) | disaggregated producer–consumer; the first version called itself elastic colocation | **32,000 concurrent environments on about 400 physical machines**; the blog claims stable operation on 10k-GPU clusters | Three controller types (**RolloutManager, SampleQueue, Trainer**) on different nodes, coordinated by RPC. Fig. 8: a **Lite-RolloutManager** feeds several **Virtual Rollout Groups, each pinned to a model version**, each with a RolloutController over ENV and LLM workers. Finished samples stream into the SampleQueue, then the Trainer. Table: **PD disaggregation + CPU KV swap**; the blog mentions efficient KV-cache reuse. **CPU-idleness-aware RPC** (§3.1.4). | streaming RPC + multiple RolloutControllers |
| **Qwen3.5** | "fully disaggregated training-inference architecture" | — | Higher hardware utilization, **dynamic load balancing**, **fine-grained failure recovery**; gradient staleness bounded by system–algorithm co-design; **3–5× end to end**. **MegaFlow:** Alibaba Cloud K8s + an Argo workflow in three stages. | Megatron + SGLang/vLLM |
| **StepFun Step 3.5** | not stated | 50k verified environments | **Steptron:** one PyTorch + Megatron-LM stack for pretraining, post-training and RL. A **Session-Router** (sticky routing of a multi-turn session; the article only names it). | Steptron |
| **Seed** | not stated | — | verl HybridFlow (single or multi-controller); Seed-Thinking **SRS** asynchronous decoupling, **3×**; MegaScale fault tolerance; verifiers isolated per process. | verl + vLLM |

**Convergent stack** (§3.1.4): Megatron-LM training + vLLM/SGLang rollout + Ray orchestration, with an in-house agentic orchestration and decoupling layer on top (slime, Forge, DORA, ASystem, Steptron + Session-Router, MegaFlow). Open-source verl, slime and AReaL are starting points, not end points.

### 1.3 M12 requirements: topology

- **Interface.** `Topology` is a config enum:
  - `colocated_sync`: on-policy baseline, used for regression tests;
  - `colocated_partial`: the **default**, as DESIGN §10.3.2 already says;
  - `disaggregated_stream`;
  - `elastic`: later.
  
  The trainer loop is identical across topologies. Only the `WeightPublisher` and the in-flight policy (§2.4) change.
- **Memory juggling knobs** for colocated mode, modelled on Kimi K3, Hy3 and MiMo:
  - `offload.train_state_on_rollout ∈ {none, cpu, nvme}`, default `cpu`: params, grads and optimizer leave the GPU during rollout.
  - `offload.kv_on_train ∈ {discard, cpu_pool}`, default `discard`, as in MiMo's `sleep_replicas()`.
  - `offload.ref_model ∈ {gpu, cpu_borrow_grad_buffer}`. The K3 trick: materialize the forward-only reference or teacher into the idle FP32 grad buffer. This is only worth building once a KL or reference term is in use; DAPO-style defaults have no KL.
- **Environments are never placed on the GPU pool.** World and sandbox workers form their own pool, sized by `env.workers`. A full GPU queue must not reduce environment concurrency, and an environment stall must not hold a GPU slot (§2.4 pausing).
- **Each run records its resource split** (rollout GPUs, train GPUs, env CPUs) and **bubble time per phase**: rollout-idle, train-idle, env-wait. This is what makes the choice between colocated and disaggregated measurable rather than ideological (art. §6 "no conclusion").

---

## 2. Sync / async modes and their exact control flows (art. §2.2–2.3, §3.2–3.3)

The article's three families (§3.2.2):
- **A, colocated async (partial rollout).** Generation and training alternate on the same GPUs. When a training batch is full, **all in-flight sequences are interrupted**, training runs, new weights load, and generation resumes. "Async" here means **generation that spans versions**, not overlap in time.
- **B, disaggregated fully streaming async.** Rollout generates continuously and finished trajectories enqueue. Training starts once a batch accumulates. During training, rollout keeps generating with the old weights; new weights are pushed every few steps. **This is true temporal overlap.**
- **C, disaggregated async + windowed sampling.** B plus a restriction: the trainer may only take finished trajectories from a sliding window in submission order.

Plus the on-policy baselines and staleness enforcement (§3.3). Below, every mode with its control flow, synchronization points, staleness bound, and what happens to in-flight work.

### 2.1 On-policy synchronous (baseline)

- **Who:** Hunyuan Hy3 open recipe (colocated sync GRPO + IcePop); GLM-5 reasoning RL; Seed1.5-VL "leans synchronous" (Table).
- **Control flow:**
  ```
  loop: generate the whole batch → wait for the slowest → reward → train → push weights
  ```
- **Synchronization points:** batch barrier, then weight push. Staleness is 0. The only off-policy-ness is the mismatch between engines (§5).
- **Cost:** the long-tail bubble. GLM-5 calls it **"substantial bubbles"**.
  - StepFun: **about 5% of samples take about 80% of generation cost**.
  - MiMo: across **25 data sources**, mean generated tokens differ **90×** and active time **66×**.
  - Qwen3-Coder-Next: SWE RL went from **50 to 130 turns** on average.

### 2.2 One-step off-policy

- **Who:** StepFun's search agent originally ran **client–server one-step off-policy**.
- **Control flow:**
  ```
  generate batch b+1 under weights v while training on batch b (generated under v−1)
  ```
- **Staleness:** exactly 1.
- **Abandoned** because the 5%/80% tail still gates each pair of steps. StepFun moved to FullyAsync for **about 10× efficiency**, staying stable at **about 20 steps of staleness** under MIS-PO (§3.2.3.2, §3.3.3.3d).

### 2.3 Colocated partial rollout (family A)

#### Kimi K3 (§3.2.3.1; Kimi k1.5 is the origin of partial rollout)

```
each round: N prompts × K samples
  when the completed fraction reaches λ ∈ (0,1), i.e. λNK trajectories: pause generation → optimize
  paused rollouts → queue; resumed first in the next round (needs sandbox pause/resume)
```
- Admits it is in an **"extreme off-policy regime"** and tolerates it with **per-token regularization** (Table: group mean + token-level log-ratio mask + "τ regularization").

#### MiMo-V2.6 (§3.2.3.1)

- "GRPO + async partial rollout, **staleness = 4**". When a batch fills, in-flight sequences are interrupted and continued next phase.
- Cost is **re-prefill**: continuation rebuilds the KV cache. So **large batches and partial rollout are chosen together** to amortize it.
- Continued trajectories do **not** recompute their rollout probabilities: the original π_rollout per token stays (§3.3.3.3b).
- Table: also "per-source expiry + scheduling suppression".

**Control flow in the open recipe (`XiaomiMiMo/verl`, read for this note):**

```python
# trainer_colocate_async.py: hooks of PPOTrainerColocateAsync
on_init_end:   checkpoint_manager.update_weights(global_steps)        # push loaded ckpt
on_train_begin: _add_async_warmup_batches(num_warmup_batches)         # pre-fill the pipeline
step():        for trigger_idx in range(parameter_sync_step):          # several updates per weight push
                   _step_once(sample_batch = train_batch_size / parameter_sync_step)
  _step_once:  on_sample_begin
               batch = replay_buffer.sample(global_steps, "train", sample_batch)   # blocks until enough groups
               on_sample_end:  abort_replicas()   # abort all unfinished requests, pause generation
                               sleep_replicas()   # discard weights AND KV cache (frees GPU for training)
               reward (if colocated RM) → balance_batch (seqlen across DP) → old_log_prob → ref → values
               → advantage → update_critic → update_actor
on_step_end:   update_weights(global_steps)        # wake replicas, load new version
               resume_generation_replicas()        # aborted requests are re-issued: re-prefill
```

`ReplayBuffer` / `ReplayBufferAsync` (`replay_buffer.py`):
- **Storage.** A KV store (TransferQueue) with keys `{uid}_{session_id}_{index}`.
- **Unit of state.** The **GRPO prompt group**, with tags `pending → running → finished | failure`. Only terminal groups (`finished`, `failure`) are sampleable.
- **Staleness bound.** `max_off_policy_threshold` = the maximum number of model versions a trajectory may **span**, `global_steps − submit_step + 1`.
- **`max_off_policy_strategy`:**
  - **`drop`:** a terminal group over the threshold is evicted and exactly `k` fresh prompts are refilled. An assertion guarantees no selected group exceeds the threshold.
  - **`wait`** ("dropless"): sampling **blocks while any pending or running prompt has reached the threshold**, so the stale prompt can finish and be trained instead of dropped.
  - `None` disables version control, so staleness is bounded only by concurrency and the produce/consume balance.
- **Selection is oldest-first** (sorted by submission step).
- **Eviction/refill matrix:** async mode evicts `k` and refills `k` for stale, DAPO-filtered (identical rewards) or failed groups alike; `num_warmup_batches` absorbs the retry cost. Sync mode with DAPO evicts `k` and refills `2k`.
- **Logged staleness metrics:**
  - `trajectory_spans` = max_v − min_v + 1;
  - `trajectory_staleness` = (step − 1) − max_v (lower bound);
  - `trajectory_staleness_worst` = (step − 1) − min_v;
  - `evicted_samples_staleness`.

#### DeepSeek-V4.1 (§3.2.3.1)

- Calls itself fully async, but because it is colocated, training still **preempts in-flight rollouts** once enough samples accumulate.
- **Sample-level scheduling:**
  - Each task has an **upper bound on in-flight samples**.
  - **"Once the number of newly completed samples reaches the GRPO group size of the next prompt, that prompt is scheduled."** Admission happens in group-sized units, triggered by sample completions, so per-task concurrency stays constant.
  - It explicitly **tried and abandoned batch-level scheduling (metrics oscillated violently)** and **prompt-level scheduling (stuck on the in-group long tail)**.
- **Pause at any token boundary:**
  - After the checkpoint switch, the rollout resumes **reusing its persisted state, no re-prefill**. One sample can span several checkpoints without paying re-prefill.
  - Table: "**token-level WAL**", a write-ahead log at token granularity.
- **Staleness:**
  - Scheduling and training-wait conditions cap the **maximum share of off-policy samples in a batch**.
  - A loss mask removes over-stale tokens (§3.3.3.3c, no formula given).
- **MoE:** **spliced routing replay**. The per-segment expert routing of a multi-checkpoint sample is concatenated rather than recomputed (§3.4.3.4).
- **Length bias:** per-dataset concurrency limits plus **dropping early-returning short samples** during the transition to steady state (§3.2.4).

#### Ring-1T C3PO++ (§3.2.3.1, §3.3.3.1)

- The round boundary is a **global token budget Φ**, not a completion ratio. Generate until the budget is spent, optimize; unfinished rollouts go to a **cross-version buffer** and continue under the next version.
- **Staleness:** each unfinished rollout has a retention period ρ(o), the number of times it has been cut. At round end ρ(o) ← ρ(o) + 1; before the next round every rollout with **ρ(o) > σ** is purged.
- In partial rollout one round = one version, so ρ equals the version lag of the oldest segment, and σ plays the same role as GLM's τ.
- **Result:** rollout phase about **2.5×** faster, end to end about **1.5×**.

#### Ling 2.6 (§3.2.4)

- **Chose partial rollout over full async *because round boundaries are deterministic*.** Generation stops on time, then train, then swap weights, which makes checkpointing and failure recovery easy.

#### The article's critique of family A (§3.2.4)

- **The interrupted trajectories are the longest and hardest, so "the more valuable the sample, the staler it gets".**
- Staleness is capped by how many times a trajectory can be cut (MiMo 4, Ring σ). Only samples finished within one round are strictly on-policy.
- Re-prefill is the second cost. DeepSeek's persisted state removes it.

### 2.4 Disaggregated fully streaming async (family B)

#### GLM-5 (§3.2.3.2)

```
inference engines generate continuously
when finished trajectories ≥ threshold: send the whole batch to the trainer
push new weights every K gradient updates
**reset the optimizer after every weight update**
```
- **Staleness, post hoc** (§3.3.3.1): each trajectory records every policy version it touched, (w_0, …, w_k) with w_0 < … < w_k. **In-flight requests keep running across weight swaps, so one trajectory spans versions.**
- Drop rule: **drop ⟺ w′ − w_0 > τ**, where w′ is the current training version.
- Worked example: τ = 3, w′ = 10. A trajectory from (6,7,8) has lag 4 > 3 and is dropped; one from (7,8,9) has lag 3 and is kept. Only the oldest version matters, because the earliest tokens deviate most.
- Table: Reasoning RL synchronous, Agentic RL fully async + optimizer reset. Environment failures are removed and **the group is backfilled (组补齐)**; heartbeat fault tolerance.

#### LongCat DORA (§3.2.3.2, Fig. 8)

- "Fully streaming multi-version async". **No batch barrier inside rollout**; execution is **per single sample** on remote workers. **2–4×** faster than synchronous (the article); the Meituan blog says 3×, "while keeping policy consistency per sample".
- Fig. 8: **Virtual Rollout Groups, one per model version**, run concurrently ("Decoupled Virtual Rollout Groups (Multi-version Models)"). Interleaved Generation / ENV / Finished bars show env time and generation time alternating per sample.
- Table's staleness entry: "SampleQueue + **new requests are dispatched only to the old version** (load 63%)". Neither the article nor the blog explains it. My reading: per-sample policy consistency is kept by pinning each sample to one version group, while older version groups drain. Treat the 63% as unexplained.

#### StepFun (§3.2.3.2)

- Moved the search agent from one-step off-policy to **FullyAsync**, about **10×**, because 5% of samples cost 80%.

#### Qwen3.5 (§3.1.3)

- Fully disaggregated; staleness bounded by system–algorithm co-design; 3–5×.

#### Seed-Thinking (Table)

- SRS asynchronous decoupling, 3×.

The article's critique (§3.2.4): highest throughput, but **staleness is no longer naturally capped by a cut count**. A separate staleness control is required (§2.6).

### 2.5 Windowed FIFO (family C), MiniMax (§3.2.3.3, Fig. 9, embedded simulator)

**Rule.**
- The generation queue holds N submitted requests in submission order T_0 … T_{N−1}.
- The training scheduler may take only **finished** trajectories inside the window [T_i, T_{i+W−1}], where T_i is the **oldest not-yet-trained** trajectory.
- Order **inside** the window is free; anything **outside** is strictly blocked, even when finished.
- The window slides only when its head is consumed.
- Typical **W = 0.3N**.

**The three regimes from Fig. 9:**
1. **Strict FIFO.** T0 is still running, so finished T1, T3, T4, T5, T7 are all blocked. Training sits idle: head-of-line (HoL) blocking.
2. **Greedy (first finished, first taken).** T1, T3, T4, T5, T7 are taken and T0, T2 keep getting pushed back. The distribution drifts toward "fast and easy" samples.
3. **Windowed, W = 4.** T1 and T3 are taken out of order. T4, T5, T7 are finished but blocked; only T0 and T2 need to finish. Once T0 is consumed the window slides to [T2, T5].

**Bound.**
- **Max Off-Policy Lag** is fixed by batch size and W together.
- The article's worked example: **batch = 8, W = 4 → out-of-order tolerance W − 1 = 3, lag bound = 8 + 3 − 1 = 10.**
- W → 1 gives strict FIFO (consistent distribution, HoL blocking). W → N gives greedy (throughput, drift toward easy samples).

**Simulator semantics** (from the TSX source; useful as an executable spec):
- `queueSize` tasks are always in flight: every pick submits one new task tagged with the current version, so concurrency is constant.
- `head` = the first task that is still running or finished-but-untrained.
- `inWindow(x) := x.id < head + W`.
- Among finished in-window tasks, pick the earliest `doneAt` (tie-break on the lower id) until the batch is full.
- Training occupies `trainTicks`; generation continues during training (disaggregated).
- On completion, version += 1.
- `lag = version_at_train − version_at_submit`.
- Reported: max lag, the share of hard tasks in the first 10 batches (early-batch easy-sample skew), and trainer idle ticks.

MiniMax also adds **MIS + PPO-style trajectory filtering** on top of CISPO in M2/M2.1 (Table, §3.5.3.2).

### 2.6 Staleness enforcement: the three routes and their exact enforcement points (art. §3.3)

| Route | Enforcement point | Rule | Who | Waste |
|---|---|---|---|---|
| **① version cap + drop** (post hoc) | when a finished trajectory is sampled | GLM: drop ⟺ w′ − w_0 > τ. Ring: purge ρ(o) > σ before each round. MiMo `drop`: span > threshold ⇒ evict + refill. | GLM, Ring-1T, MiMo (open) | highest: already-generated work is thrown away, and the long, hard trajectories are the ones that cross the line |
| **② staleness-manager admission gating** (a priori) | when a new rollout is **admitted** | Ling 2.6: every rollout **segment** is tagged with the engine's policy version at generation, and **\|P_in-flight\| ≤ S × B** (S = `max_staleness`, B = `consumer_batch_size`). The trainer takes B per step, so a new rollout waits about ≤ S steps. Segments over the version bound are also **retired**. | Ling 2.6 (with sub-second broadcast) | lowest. Ring-2.6's effective staleness is "a few optimizer steps", with "no loss in stability or final reward vs synchronous". Needs fine version tags and fast weight broadcast. |
| **③ algorithmic mask / truncation** | in the **loss**, per token or trajectory | DIS, MiMo four-bound, MIS-PO, HisPO, DeepSeek stale-token mask (§5) | GLM, MiMo, StepFun, LongCat, DeepSeek | tokens lose gradient. **The most mainstream route**, because exact per-token π_old is infeasible under multi-version async (GLM, SAO). |

Further mechanisms:
- **MiMo `wait`:** route ① made dropless. The *sampling* barrier blocks until stale in-flight work finishes.
- **DeepSeek:** a *batch composition* constraint (max off-policy share), plus a mask.
- **The open problem (§6):** no consensus on the staleness cap. MiMo uses 4, StepFun observed stability at about 20; nearly everyone else is unpublished.

### 2.7 What happens to in-flight requests at a weight update (cross-lab synthesis)

| Policy | Mechanism | Who | Version purity of a single model call |
|---|---|---|---|
| **abort + re-issue** | abort, then sleep (discard weights and KV), then update, then resume; continuation re-prefills | MiMo (open `colocate_async`) | each continuation chunk has one version; a call can be split across versions |
| **pause into queue, resume first** | paused rollouts resumed with priority next round; sandbox paused | Kimi K3 | the same |
| **pause at a token boundary with persisted state** | resume without re-prefill (token-level WAL) | DeepSeek-V4.1 | a call spans versions at a token boundary; routing spliced |
| **keep running across the swap** | the engine swaps weights under running requests; the trajectory records (w_0..w_k) | GLM-5 (disaggregated) | spans versions (the granularity is not stated) |
| **pin each sample to a version group** | per-sample policy consistency; old groups drain | LongCat DORA | one version per sample |
| **budget cut + cross-version buffer** | unfinished rollouts continue under the next version | Ring-1T C3PO++ | spans versions |

### 2.8 M12 requirements: async modes, granularity of synchronization, staleness

#### Version semantics

- **Version.** `policy_version` is an integer incremented by the `WeightPublisher` each time a new weight set becomes servable.
  - **Default K = 1**: one publish per optimizer step. The knob is `publish_every_k_updates`, GLM's K and MiMo's `parameter_sync_step`.
  - Optional `reset_optimizer_on_publish` (GLM), default `false`.
- **Version-switch granularity.** Default: **at model-call boundaries only.** A `Turn` (one model call) is always served by exactly one version, which keeps the existing `Turn.policy_version: int` exact.
  - In colocated mode, at the sample barrier an in-flight call is aborted and **re-issued whole** under the new version. MiMo-style: the partial completion is discarded, not stitched. This wastes at most one call's tokens, but needs no per-token versions.
  - `version_switch = "token"` (DeepSeek-style continuation, later) requires a new `Turn.version_spans: list[[start_token_idx, version]]` and per-span routing. **Until that field exists, stitching a completion across versions is forbidden.** The rollout record must be able to prove version purity.
- **The world never pauses for weights.** Environment steps (tool execution, verifier calls) continue during weight sync; only model calls wait. A rollout whose next model call is pending holds **no GPU state**. Its World is checkpointable (`runtime/checkpoint.py`), equivalent to Kimi's "sandbox pause" without a microVM.

#### Modes and synchronization points

| Mode | S1: sample barrier | S2: weight publish | S3: in-flight at publish |
|---|---|---|---|
| `colocated_sync` | wait for all N·n rollouts of the batch | after every train step | none in flight |
| `colocated_partial` (**default**) | the buffer holds ≥ B terminal groups that pass the staleness check (`wait` or `drop`) | after K updates; generation resumes after publish | abort in-flight calls, discard KV, re-issue under the new version; Worlds keep their state |
| `disaggregated_stream` | the same buffer rule; generation never stops | after K updates; the publisher swaps the engine at **call boundaries** (drains in-flight calls, or routes new calls to the new version: DORA-style version groups) | in-flight calls finish on their version |
| `windowed_fifo` (a buffer policy, combinable with either async mode) | take only terminal items whose submission index < head + W; W = `fifo_window_frac` × N, **default 0.3** | as the host mode | as the host mode |

#### Staleness enforcement (all three routes, layered)

- **Admission:** `max_inflight = S × B` (Ling), with `S = max_staleness`, **default 4** (MiMo's published value).
- **Sampling:** `staleness(r) = v_train − r.oldest_version()` (already in `RolloutRecord`).
  - `staleness_strategy ∈ {wait, drop}`, **default `wait`.** Long-horizon rollouts are the most expensive and the most likely to cross the line (art. §3.3.4), and dropping them biases toward short tasks.
  - `drop` evicts and refills the same number of groups.
- **Loss:** the selected mismatch/staleness correction (§5.4).
- **Batch composition (optional):** `max_offpolicy_frac` (DeepSeek), the share of batch samples with staleness ≥ 1.

#### Logged every step

- `trajectory_spans` mean and max;
- `staleness` mean and max, plus worst (oldest version);
- `evicted_staleness`;
- `admission_blocked_s`;
- `sample_barrier_wait_s`;
- `reprefill_tokens` (tokens recomputed because of aborts);
- the share of on-policy samples.

---

## 3. Parallelization granularity, long tail, length bias, balancing (art. §3.2, §3.5, Table)

### 3.1 Dispatch granularity

| Granularity | Who | Verdict in source |
|---|---|---|
| **batch-level** | — | DeepSeek-V4.1 tried it: **"metrics oscillated violently"** |
| **prompt / group-level** (schedule and admit whole GRPO groups) | — | DeepSeek-V4.1 tried it: **"stuck on the in-group long tail"**. SAO (GLM): group sampling causes **"latency-driven off-policy"**, because the group waits for its slowest member, and it is incompatible with environments that give one trajectory per prompt (§3.5.1 pathology 3). |
| **sample-level with group-sized admission** | DeepSeek-V4.1 | per-task in-flight cap; schedule the next prompt when the newly completed count ≥ its group size |
| **single-sample streaming** | LongCat DORA | no batch barrier; each sample executes on remote workers |
| **group terminal-state units in the buffer** | MiMo (open) | groups are the sampling unit; sessions inside run independently |
| **completion-ratio rounds** | Kimi K3 (λNK) | — |
| **token-budget rounds** | Ring-1T C3PO++ (Φ) | — |
| **group size = 1** | GLM SAO | removes the group barrier entirely; needs a critic |

**Environment-step granularity:**
- Rollouts alternate generation and environment segments (Fig. 8 bars).
- Kimi AgentENV **pauses the microVM while it waits for inference**, which is up to **98% of its lifecycle** (§3.8.3). The environment step is the unit at which sandbox resources are released.

### 3.2 Training-unit granularity

| Unit | Who | Note |
|---|---|---|
| token, global token-mean | DAPO, CISPO (denominator Σ\|o_i\| over the group, in practice the global batch) | removes the sequence-length dilution |
| token-mean per sequence, then mean over sequences | original GRPO, IcePop (1/G · 1/\|y_i\|) | long sequences get diluted |
| prompt-mean | MiMo: token-average inside each response, then average across prompts, against length inflation | — |
| sequence (one ratio per trajectory) | GSPO | clipping happens per whole sequence |
| global constant G·max_i \|y_i\| | HisPO | against length bias |
| (s_t, a_t) pair | MiniMax M2 CISPO | the atomic unit; the policy need not know whether s_t came from appending, truncating or rewriting history (context management); credit stays episode-level |
| segment (compaction) with token-level normalization | CompactionRL | removes the bias from the number of segments |
| turn-level token penalties | Qwen3-Coder-Next | penalty only on the tokens of illegal tool calls |
| sub-groups by reasoning effort | DeepSeek (§3.5.4) | "split groups by effort" |

The guiding principle (§3.5.4, from GSPO): **the optimization unit should match the reward unit.**

### 3.3 Long tail and length bias: who says what

**Numbers:**
- StepFun: 5% of samples, 80% of generation.
- MiMo: 90× spread in tokens, 66× spread in active time across 25 sources.
- Qwen3-Coder-Next: 50 → 130 turns.
- Kimi K3: 1M context.
- MiniMax: up to 200k context.

**Length bias** (§3.2.1, §3.2.4): in any first-finished-first-batched streaming scheduler, short samples arrive first, early batches are dominated by them, and the model drifts toward short outputs. Remedies:
- **DeepSeek:** per-dataset concurrency limits, which indirectly control source mix, and **dropping early-returning short samples** so the batch length distribution moves smoothly to steady state.
- **MiniMax windowed FIFO:** bounds how far fast samples can overtake.
- **MiMo `num_warmup_batches`:** pre-fills the pipeline before training so early batches are not just the fastest returns.
- **HisPO / DAPO / MiMo:** loss-side length normalization (table above).

**Overlong handling:**
- DAPO overlong filtering (+6 points) plus soft punishment:
  R = 0 if |y| ≤ L_max − L_cache; ((L_max − L_cache) − |y|)/L_cache in between; −1 beyond L_max. Values: L_max = 20,480, L_cache = 4,096.
- StepFun **truncation-aware value bootstrap** R̂ = V_φ(s_T) for truncated trajectories, stable at **20% truncation**.
- Qwen penalizes unfinished trajectories (max turns exceeded).

### 3.4 Load balancing and packing

- **MiMo open recipe:** `_balance_batch` balances sequence length across DP ranks before each update. Fig. 6 shows a **Packer** and a **Sample Mixer** in the control plane.
- **MiniMax Prefix Tree Merging (40×)** (Forge blog):
  - Multi-turn completions that share prefixes are **merged into one tree**.
  - Attention primitives keep the forward logically equivalent; the tree is then split back for the standard loss.
  - **"Zero impact on loss computation or metrics."** It removes recomputation of prefixes shared across turns and across group members.
- **Qwen3.5:** dynamic load balancing.
- **LongCat:** CPU-idleness-aware RPC.
- **Forge:** cost-aware KV scheduler (queue delay vs cache migration).

### 3.5 M12 requirements: granularity

- **Dispatch unit = one rollout (sample).** Groups are an admission and advantage construct, never a dispatch barrier.
  - Admission follows DeepSeek's rule: per task family, `inflight_cap`; when `free_slots ≥ n`, admit the next group of n.
  - GRPO advantage is computed only when a group is terminal.
  - In critic mode (n = 1) groups vanish.
- **Training units are selectable:**
  - `loss_agg ∈ {token_mean_global (default, DAPO/CISPO), seq_mean_token_mean, prompt_mean (MiMo), seq_ratio (GSPO), const_max_len (HisPO)}`;
  - `credit_unit ∈ {sequence, turn (M6 turn advantage on that call's tokens, already in Turn.advantage), segment}`.
- **Length-bias controls:**
  - per-source `inflight_cap`;
  - `warmup_batches` (default 1 for async modes);
  - optional `drop_early_short`, which drops returns shorter than the p-th length percentile during the first M steps;
  - the logged metric is **the gap between the batch length distribution and the steady-state length distribution**: a KS statistic per step.
- **Packing:**
  - sequence-length balancing across DP ranks (required);
  - prefix sharing for GRPO groups and multi-turn calls (optional; Forge's tree merging, or plain packing with shared-prefix attention).
  
  Crucible's linear delta-token storage (`Turn.prompt_is_delta`) already holds the tree structure needed to reconstruct shared prefixes.

---

## 4. Weight sync, rollout engines, KV reuse, sandboxes (art. §3.1.4, §3.4, §3.8, Table, Forge blog)

### 4.1 Weight sync: mechanisms and costs

| Mechanism | Cost | Who |
|---|---|---|
| **AState:** zero-redundancy P2P; protocol chosen dynamically by NUMA topology (RDMA / NCCL / shared memory) | **1T params in < 10 s**, sub-second updates | Ant ASystem |
| slime as AState criticizes it: **destroys and rebuilds NCCL groups** | "minutes" | GLM slime (per Ant) |
| NFS / NCCL checkpoint-file sync | "minute-level" (per Ant) | generic |
| sleep → discard weights and KV → `update_weights` → wake → resume (in-process, colocated) | re-prefill of aborted requests | MiMo open recipe |
| **QDQ weights:** after each update, quantize then dequantize expert weights under the rollout MXFP4 kernel's numerics, then push | — | MiMo (Fig. 6 edge "QDQ weights") |
| router-level service lifecycle + TITO gateway | — | GLM slime |
| a push every K gradient updates (+ optimizer reset) | amortizes sync cost | GLM |

The article's claim (§3.1.4): **hot parameter update is the choke point for both topologies**, and especially for disaggregated ones. Sub-second broadcast is what makes Ling's tight admission gating viable.

### 4.2 Rollout engine details

- **Engines:** SGLang or vLLM everywhere. TITO is mandatory for the correction math:
  - Kimi K2.5 provides a **custom RL-facing inference API** that records all output logprobs;
  - GLM uses a **TITO gateway**;
  - a detokenize → retokenize round trip misaligns token boundaries, after which "no mask, however clever, can be applied" (§3.4.4).
- **PD disaggregation:** slime, Forge (heterogeneous), LongCat (with **CPU KV swap**).
- **KV and prefix reuse across turns:**
  - Forge **global L3 KV pool on a DFS** with a cost-aware scheduler;
  - MiMo **Context Cache + L2Cache** (Fig. 6);
  - LongCat KV reuse and CPU KV swap;
  - DeepSeek persisted per-token state (no re-prefill on resume);
  - Kimi K3 CPU-DRAM KV pool between rounds.
  
  The article does **not** mention radix or prefix caching by name. Multi-turn KV reuse shows up through these components.
- **Speculative decoding:** Forge MTP heads, fine-tuned online against the moving policy (Top-K KL). MiMo decodes with "DFlash" (Fig. 6).
- **Captured per token** for the corrections:
  - logprobs (everyone with a mask);
  - MoE expert ids;
  - the **top-p candidate set** (MiMo R3: sampling renormalizes inside the candidate set, so training must renormalize logprobs over the same set). Implementation: one **fixed-shape, full-vocab bitmap** dense transfer GPU → CPU (no host sync, no truncation), sparse afterwards. At top-p 0.97 the candidate set averages **fewer than 5 tokens**, so the overhead is negligible.
- **Session routing:** StepFun Session-Router; slime router.

### 4.3 Sandboxes and environment services

| System | Substrate | Numbers |
|---|---|---|
| **Kimi K3 AgentENV** | Firecracker microVM; **pause** (no resources while waiting for inference, up to 98% of lifecycle), **fork** (side-effect-free reward judging), **snapshot** (error recovery) | **incremental checkpoint / resume latency 133 ms / 49 ms**; **6.5× memory oversubscription**; **51,219,741 sandboxes and 1,505,678 images** created over training and evaluation. Earlier, ordinary containers hit kernel panics and deadlocks. |
| **DeepSeek DSec** | four substrates: Function Call / Container / microVM (Firecracker) / full VM (QEMU); a **custom placement engine, not Kubernetes** | hundreds of thousands (V4) to **millions (V4.1)** of concurrent sandboxes; **sub-NUMA placement raised per-node containers from about 1,000 to 2,500+**; a per-sandbox AppArmor profile + eBPF network policy |
| **Ling ASandbox** | lightweight, serverless, on demand (HTTP and MCP) | **100 ms start, 5,000 QPS at 200 ms latency**; thousands of replicas; trades isolation depth for throughput |
| **LongCat DORA** | remote env workers over RPC | **32k concurrent environments, about 400 machines** |
| **GLM-5** | per-task microservices | 10k+ SWE environments (9 languages); thousands of Terminal environments (Docker build accuracy > 90%); a Web Knowledge Graph of 2M+ pages |
| **Qwen3-Coder-Next** | Docker images | 807,693 real-PR instances (52,960 repos) + 851,898 synthetic-bug tasks |
| **MiMo** | local mocks + container network isolation | 7k+ environments; reference-patch tasks need 8 consistent reruns |
| **StepFun** | — | 50k verified environments (15k+ repos, 20+ languages); environment building is itself trained by RL (about 40% success) |
| **Ling 2.6** | Claude Code + MCP auto-builds | about 220,000 Docker images |
| **MiniMax** | — | "hundreds of thousands" of RL environments |

**Noise handling relevant to the scheduler:**
- MiMo's **Penalty Module** masks infra failures out of the loss.
- DeepSeek treats an **environment crash as a failed trajectory** with a "recoil" signal.
- GLM removes environment failures and **backfills the group**; heartbeat fault tolerance.
- **Anti-hack interception:**
  - GLM-5.2 **intercepts a detected hack call online and returns dummy information instead of aborting**, because sudden aborts destabilized async training.
  - Qwen intercepts any call containing both `github.com/{repo}` and `git`/`curl`/`wget`.

### 4.4 M12 requirements: weights, engines, sandboxes

- **`WeightPublisher`** interface: `publish(state_dict_ref) -> version`, `serving_version() -> int`, `latency_s` metric. Implementations in order:
  1. `inproc_colocated`: sleep/wake plus in-place load, the verl/MiMo pattern; default;
  2. `checkpoint_reload`: disk safetensors plus engine reload;
  3. `nccl_broadcast`: disaggregated, later.
  
  The publish is **atomic per call**: an engine response carries the version that produced it, and the capture proxy writes it into `Turn.policy_version`. **The trainer never infers the version from wall time.**
- An optional **`post_update_transform`** hook runs before publish, e.g. MiMo-style QDQ to the rollout quantization grid. It is a no-op for BF16 rollout.
- **Engine capture contract** (extends the existing proxy):
  - token ids in and out;
  - per-token behaviour logprobs;
  - version;
  - optionally MoE routing ids (the existing `Turn.routing`);
  - optionally the **top-p candidate set**, as a new `Turn.candidates` sparse field, needed only if the sampling uses top-p < 1 and the correction renormalizes.
  
  A detok→retok round trip anywhere is a schema violation (`prefix_break`, already masked).
- **Prefix and KV reuse** for multi-turn calls is an engine configuration (prefix caching on), validated by a metric: `cached_prefix_tokens / prompt_tokens` per call. With colocated `discard`, the cache resets at every publish; the metric makes that cost visible.
- **Sandboxes:** reuse M1/M10. M12 requires of them:
  - `env.workers` independent of GPU;
  - heartbeat or timeout per environment step, giving infra failure → `mask_sample` (already in the schema);
  - an agent-caused crash → failure, never masked;
  - **group backfill** policy for infra-masked members ∈ {pad (MiMo default), refill_member, drop_group};
  - side-effect-free judging on a forked snapshot (Kimi fork), where the World backend supports it.
- **In-loop anti-hack** at the gateway: a detected shortcut probe gets refusal or dummy content and the rollout continues (GLM-5.2, consistent with Crucible P27). The scheduler must not abort the episode for a *blocked* probe.

---

## 5. Train–inference mismatch: sources and every correction (art. §2.4, §3.3.3.3, §3.4, §3.5)

### 5.1 Notation (art. §3.3.3.3)

- π_θ: the current policy on the training engine.
- π_θold: the version that generated the token, **computed on the training engine**.
- π_rollout: the same version's probability **on the inference engine** at sampling time. Other papers write μ, μ_θold or π_vllm.
- The true ratio factors as (HisPO):

  r_t = π_θ / π_rollout = **r_t^dis** · **r_t^stale**, with r^dis = π_θold / π_rollout (engine mismatch, same weights) and r^stale = π_θ / π_θold (policy drift).

### 5.2 Sources (§3.4.1)

1. **Different kernels, precision and reduction order** in the two engines. Even bit-identical weights give different logprobs.
2. **MoE routing flips:**
   - top-k gating is discrete, so a perturbation of 1e-3 can swap the experts;
   - Qwen measured on **Qwen3-30B-A3B-Base (48 layers): about 10% of activated experts change per RL update** for the same rollout, worse in deeper layers;
   - this makes token-level IS ratios meaningless.
3. **Self-amplification** (Ring-1T, compound probability discrepancy theorem): with δ_t = D_KL(π_infer(·;θ_t) ‖ π_train(·;θ_t)), there is η > 0 such that δ_{t+1} ≥ (1 + ημ/2) δ_t ⇒ δ_t ≥ (1 + ημ/2)^t δ_0, a **lower bound**. It shows up as gradient-norm spikes, entropy collapse and reward collapse.
4. **Low-precision rollout** (FP8/INT8), and **LM-head precision**. MiniMax saw "segmented horizontal lines" in a train-vs-infer probability scatter on M1; the cause was LM-head precision.
5. **Non-deterministic operators.** GLM's DSA indexer top-k in SGLang.
6. **Top-p renormalization.** If the candidate set is not replayed, training normalizes over the full vocabulary (MiMo).
7. **Tokenization round trips** (§3.4.4).
8. The table adds **Hunyuan freezing the MoE expert bias**.

### 5.3 Corrections, with formulas and where each is used

**Group 1: algorithmic mask / truncation.**

- **TIS** (Yao et al.; the de facto baseline in slime, verl and OpenRLHF; §3.4.3.1a):

  E_{a∼π_infer}[ min(π_train/π_infer, C) · R(a) · ∇ log π_train(a) ]

  One-sided truncation of r^dis. It beats PPO-IS (π_infer as the PPO denominator) and vanilla IS, and the gap grows with mismatch, so it is what **makes low-precision rollout viable**. It gains nothing when mismatch is small; under strict on-policy the ratio is 1.
- **IcePop** (Ring-1T; §3.4.3.1b):

  J = E[ (1/G) Σ_i (1/|y_i|) Σ_t ( M(π_train(y;θold)/π_infer(y;θold); α, β) · min(r_{i,t}Â, clip(r_{i,t}, 1−ε, 1+ε)Â) − γ D_KL ) ]

  with r_{i,t} = π_train(θ)/π_train(θold) and **M(k) = k if k ∈ [α, β], else 0**.
  - **The gate uses r^dis; the PPO term uses r^stale.**
  - In range, the weight becomes r^dis · r^stale = full r ("calibration"); out of range it is masked.
  - **α = 0.5, β = 5.** KL coefficient 0.0 in Ring-1T's GRPO.
  - vs TIS: two-sided, and discard rather than suppress. On Ring-mini-2.0 it was about 6% better on AIME25.
- **KPop** (Ling 2.6; §3.4.3.1c): **binary KL instead of a ratio**, because fixed ratios over-mask low-probability tokens.

  D^B_KL(p‖q) = p log(p/q) + (1−p) log((1−p)/(1−q)), with p = π_train(y_t), q = π_infer(y_t)

  M_KPop = 𝟙[D^B(π_train‖π_infer) ≤ φ] · 𝟙[D^B(π_infer‖π_train) ≤ φ]

  - The article's worked examples:
    - p = 5e−5, q = 1e−5 (ratio 5): D^B = 8.047e−5 − 4.000e−5 = **4.05e−5**, kept.
    - p = 0.9, q = 0.45 (ratio 2): 0.6238 − 0.1705 = **0.4534**, masked.
  - Small-probability approximation: D^B ≈ p ln(p/q) − (p − q).
  - One hyperparameter, φ. **SWE-bench Verified 70.8% → 76.28%.**
- **Kimi K2.5/K3:** token-level **log-ratio mask**; inside [α, β] normal gradient, outside zero, independent of the advantage sign. Requires strict TITO with all logprobs recorded.
- **Hunyuan Hy3:** IcePop-style token-level rollout-IS switch with **thresholds [0.5, 4.0]**; out-of-range weight 0; measured **`rollout_probs_diff` < 0.015 throughout**.
- **GLM DIS** (Direct Double-sided IS; §3.3.3.3a, also SAO's objective):

  r_t = exp(log π_θ − log π_rollout)

  L = E_t[ f(r_t; ε_ℓ, ε_h) · Â_t · log π_θ ], with f(x) = x if 1−ε_ℓ < x < 1+ε_h, else 0

  - **No π_θold**: no old-policy forward and no version history.
  - Masks both sides regardless of the sign of Â, stricter than PPO.
  - SAO values: **math TIR ε_ℓ = 0.3, ε_h = 5.0; coding agent ε_ℓ = 0.8, ε_h = 3.0.**
  - It stacks on other methods: SAO's table has GRPO 84.2/54.8 → **+DIS 93.5/70.8** (AIME2025 / BeyondAIME); vanilla VAPO without DIS collapses.
- **MiMo four bounds by advantage sign** (§3.3.3.3b):

  r = sg[π_θ/π_rollout]

  M = 𝟙[(Â ≥ 0 ∧ ε⁺_l ≤ r ≤ ε⁺_h) ∨ (Â < 0 ∧ ε⁻_l ≤ r ≤ ε⁻_h)]

  L_{i,t} = r · M · Â_i · log π_θ, aggregated prompt-mean

  - The bounds are on the ratio itself. **Both pairs start at [0.2, 5.0]**, symmetric in log space (±1.61).
  - **Entropy-adaptive:** when entropy is too low, widen the positive bounds and tighten the negative ones; the reverse when it is too high.
  - Partial-rollout continuations keep their original π_rollout.
- **DeepSeek-V4.1:** a stale-token mask plus a cap on the off-policy share of a batch (no formula).
- **StepFun MIS-PO** (§3.3.3.3d, §3.5.3.5):

  x_t = π_θold / π_rollout, ρ̄(τ) = (Π_t x_t)^{1/T}, 𝕀(x) = 𝟙[ρ_min ≤ x ≤ ρ_max]

  L = −E[ 𝕀(x_t) · 𝕀(ρ̄(τ)) · log π_θ · Â_t ]

  - **Token band [0.5, 2]; trajectory band [0.996, 1.001]**, i.e. −0.004 ≤ mean ln x_t ≤ +0.001 (asymmetric, unexplained).
  - Kept tokens have weight 1 ("effectively on-policy"): lower variance.
  - The band bounds the *mean per-token* bias: at the lower edge the sequence ratio e^{−0.004T} is 0.67 (T = 100), 0.018 (T = 1000) and 2e−9 (T = 5000).
  - **Stable at about 20 steps of staleness** (search agent).
  - Not a real Metropolis–Hastings test: acceptance is deterministic.
- **LongCat HisPO** (§3.3.3.3e):

  J = E[ (1/(G · max_i|y_i|)) Σ_i Σ_t H_{i,t} · min(r Â, clip(r) Â) ]

  H_{i,t} = 𝕀{|exp((1/|y_i|) Σ_j log r^dis_{i,j}) − 1| < δ_seq} · 𝕀{|r^dis_{i,t} − 1| < δ_tok}

  - Layers 1 and 2 gate on r^dis only, first per sequence (geometric mean), then per token.
  - Layer 3 clips the full r = r^dis · r^stale with **three bounds**: ε^neg_low and ε^neg_high for negative-advantage tokens (dual clip; this closes the unbounded r·Â as r → ∞ when Â < 0, which MoE routing changes provoke), and ε^pos_high for positive ones.
  - The denominator G · max|y| is a global constant against length bias.
- **CISPO** (MiniMax; §3.5.3.2), which clips the IS **weight**, not the update:

  J = E[ (1/Σ_i|o_i|) Σ_i Σ_t sg(r̂_{i,t}) Â_{i,t} log π_θ ], with r̂ = clip(r, 1−ε^IS_low, 1+ε^IS_high) and r = π_θ/π_θold

  - **M1** uses the two-sided [1−ε_low, 1+ε_high]; **M2** uses **[0, 1+ε_high]**, which allows aggressive down-weighting of actions that became impossible.
  - sg is mandatory: otherwise the gradient gains a factor (1 + log π_θ).
  - Unified form with a token mask M_{i,t} (PPO's implicit mask; M ≡ 1 is pure CISPO).
  - No KL. M2/M2.1 add MIS + PPO-style trajectory filtering.

**Group 2: sequence-level IS bypass.**

- **GSPO** (Qwen; §3.4.3.2):

  s_i = (π_θ(y_i|x)/π_θold(y_i|x))^{1/|y_i|}

  J = E[ (1/G) Σ_i min(s_i Â_i, clip(s_i, 1−ε, 1+ε) Â_i) ], Â from group (r − mean)/std

  - **Clip range 3e−4 / 4e−4** (GRPO uses 0.2 / 0.27).
  - About **15%** of tokens are clipped vs GRPO's 0.13%, yet GSPO is more sample-efficient.
  - **Removes the need for routing replay.** Before GSPO, GRPO MoE training only converged with Routing Replay: cache the experts activated under π_θold and replay them under π_θ, at extra memory and communication cost and with capped MoE capacity.
  - It might allow using inference-engine likelihoods directly ("intuitively", untested).
  - **Counter-evidence:** StepFun found that the GSPO train/inference density ratio π_θold/π_rollout **keeps diverging** on MoE (plateau at about 200 iterations), while MIS-PO keeps it bounded.
  - Users: LongCat, Ling 2.6 agentic specialists.

**Group 3: eliminate the bias at the system level.**

- **DeepSeek-V4:** bitwise alignment of pretraining, post-training and inference as a design goal.
  - Batch-invariant attention with a **dual-kernel strategy**: full waves use one SM per sequence; the last partial wave uses multiple SMs with the same accumulation order.
  - **DeepGEMM replaces cuBLAS, with no split-k.**
  - Attention backward uses a per-SM accumulation buffer plus a deterministic global sum.
  - Rollout and serving use **native FP4** weights.
- **Shared quantization:**
  - **MiMo QDQ:** after every update, quantize-dequantize the expert weights to the rollout MXFP4 kernel's constraints, so both engines see bit-identical experts.
  - **Kimi K3 MXFP4 QAT from SFT through RL** "eliminates train–inference mismatch". The table says MXFP4/MXFP8 QAT.
- **FP32 LM head:** MiniMax (M1) and Ling 2.6 found it independently.
  - Ling: Megatron and SGLang both compute the LM head in FP32, **+2 reward points**. The path is **BF16 input + FP32 output**: it aligns output activation precision, not weight precision.
  - The article calls this **the highest-value single change**.
- **GLM:** deterministic `torch.topk` replaces SGLang's non-deterministic CUDA top-k, and the **DSA indexer is frozen**. Indexer replay (k = 2048) is impractical.
- **Ling:** module-aware FP8 (Table).

**Group 4: forced routing consistency.**

- **R3 (Rollout Routing Replay, MiMo):** record the inference engine's expert indices per token and replay them in training, plus replay of the top-p candidate set; overhead negligible. It removes the routing-flip symptom only; numeric differences remain.
- **Naming:** the article never says "R2". The "Routing Replay" it attributes to the pre-GSPO Qwen setup (replay π_θold's training-side routing under π_θ) is what others call R2. R3 replays the **rollout** engine's routing.
- **DeepSeek-V4.1 spliced routing replay:** for samples generated across checkpoints, concatenate each segment's routing ("incremental R3").
- **StepFun Routing Confidence Σ_k:** the mean probability mass of the activated experts.
  - Low Σ_k amplifies mismatch. There is an approximate **phase transition**: low-confidence models need router replay or strict on-policy; high-confidence models train off-policy without intervention.
  - **"Measure before you decide whether you need R3."**

**Layering in practice.** MiMo uses QDQ (group 3) + R3 (group 4) + the four-bound mask (group 1) together. Ring-1T admits IcePop does not reach perfect consistency.

### 5.4 M12 requirements: mismatch

- **Always compute and log `rollout_probs_diff`** (mean |π_train − π_rollout| per token), `r_dis` quantiles, the masked-token share and the clipped-token share.
  - When there is no correction, r^dis needs one forward of the training engine over the rollout tokens with the behaviour version's weights. In colocated K = 1 mode that **is** π_θold (the pre-update forward MiMo's `_compute_old_log_prob` does), so it costs one forward per step.
- **`mismatch.correction` menu**, each a pure loss-side function over (logp_train, logp_old?, logp_rollout, Â, masks):
  - `none`;
  - `tis(C)`;
  - `icepop(α=0.5, β=5)`;
  - `kpop(φ)`;
  - `dis(ε_ℓ, ε_h)`;
  - `mimo4(bounds, entropy_adapt)`;
  - `mispo(tok=[0.5,2], traj=[0.996,1.001])`;
  - `hispo(δ_seq, δ_tok, ε^neg_low, ε^neg_high, ε^pos_high)`;
  - `cispo(ε_low, ε_high)`;
  - `gspo(ε=3e−4/4e−4)`.
  
  **Default: `dis` with the SAO coding-agent bounds ε_ℓ = 0.8, ε_h = 3.0.** It needs no π_θold, so it is valid under multi-version async with call-level versions. It is the ingredient with the largest measured gain in SAO, and it is one line of loss.
- **Golden tests** reproduce the article's numbers:
  - KPop: 4.05e−5 kept and 0.4534 masked at φ between them;
  - MIS-PO trajectory filter: geometric mean 0.995 rejected, 0.999 accepted;
  - IcePop: r^dis = 6 masked, r^dis = 0.7 calibrated.
- **`lm_head_fp32: true` by default**, in both engines where configurable; the cheapest fix.
- **Routing replay** applies only when the policy is MoE. M12 measures Routing Confidence Σ_k first and logs it; `routing_replay ∈ {off, r3}` requires `Turn.routing` for every completion token.
- **Top-p candidate replay** is required whenever rollout top_p < 1 *and* the correction uses π_rollout as a denominator. Otherwise run rollouts at top_p = 1.0, the **default**, to avoid needing the field.

---

## 6. Data pools, middleware, multi-agent, compaction, harness abstraction (art. §3.1, §3.5.3.6, §3.6.3.4, §3.8, Forge blog)

### 6.1 Middleware and data pools

**MiniMax Forge** (§3.1.3, Fig. 7, blog):
- Three decoupled modules talk through standard interfaces, so agent and training loop scale independently:
  - the **Agent side**, either a black box (an opaque agent calling an API) or a white box (an agent loop with an LLM server, an Env server and a Reward server);
  - the **Middleware**;
  - the **Engines** (Rollout ↔ "sync weights" ↔ Train).
- **Gateway Server:** a standardized completion gateway between agent and LLM over standard protocols, isolating model complexity from agent logic. The token capture point.
- **Data Pool:** distributed storage that asynchronously collects trajectories and reports. Records = *Completions* {prompt_ids, response_ids, …} + *Rewards* {outcome, process}. It is the buffer that decouples generation from training.
- The article says Forge plugs in any white- or black-box scaffold through **"four interfaces"** but does **not** name them; neither does the Forge blog.
- Context management for black boxes includes aggressive strategies such as "Truncate BC".

**GLM-5 slime:** a Multi-Task Rollout Orchestrator with **per-task microservices** (rollout + reward), 1k+ concurrent rollouts, heartbeat fault tolerance, a TITO gateway.

**LongCat:** SampleQueue between RolloutManager and Trainer.

**MiMo:** "Payload Porter" (Sample Mixer, Packer, Distributed Trajectory Storage). In the open code, a TransferQueue KV store with tag (metadata) and value (payload) separated.

**Kimi:** Rollout Manager + Env Pool (black-box via LLM Gateway, white-box), pluggable Toolset / Judge / Prompt Enhancement (Fig. 4).

**Asynchronous reward.** MiMo GAR grading runs **asynchronously; if the grader output is unavailable, training falls back to the original advantage** (§3.6.3.2c). So the grader is off the trainer's critical path, behind a deadline.

### 6.2 Multi-agent DAG and critical path

- **DeepSeek-V4.1 Agent Team:**
  - The events of a multi-agent execution and their collaboration dependencies form a **DAG**.
  - The **critical-path length** (not the event count) is the "derived latency" penalty.
  - Subtasks off the critical path cost nothing extra.
- **Kimi K2.5 PARL:**
  - **Subagents frozen**; their trajectories are excluded from the objective and only the orchestrator is trained.
  - r = λ₁ r_parallel + λ₂ r_finish + r_perf, with λ₁, λ₂ annealed to 0.
  - **CriticalSteps = Σ_t (S^(t)_main + max_i S^(t)_sub,i)**.
  - Wide search: latency down up to 4.5×, item F1 72.8 → 79.0%.
- **Kimi Agent Swarm as a context manager:** subagents hold bounded local contexts and return only relevant output ("context sharding"). It beats Discard-all on BrowseComp.

### 6.3 Context compaction in RL

**CompactionRL** (Tsinghua; used in GLM-5.2; §3.5.3.6, §3.6.3.4, §3.8.3). The exact recipe:

1. **Trigger:** remaining context budget < **10,240 tokens**.
2. **Who summarizes:** **the same trainable policy** generates the summary.
3. **Rebuild:** a new context from the summary plus the **most recent k = 2 steps**.
4. A rollout therefore becomes a variable number of **segments** σ_1..σ_S. They share one task-level reward, with different lengths and different distances to the outcome.
5. **The summary segments are in the RL objective**, jointly trained with the trajectory under the **same task reward**. There is **deliberately no separate summary-quality reward**.
6. **Algorithm:** PPO + critic. Group normalization fails with a variable number of segments.
7. **Token-level loss normalization:** loss over generated tokens, not samples or segments. A rollout that compacted more often does not get extra weight.
8. **Cross-trajectory GAE:**
   - local GAE A^loc_{s,i} = Σ_ℓ (γλ)^ℓ δ_{s,i+ℓ}, with δ_{s,i} = r_{s,i} + γV(x_{s,i+1}) − V(x_{s,i});
   - then **Â_{s,i} = (γλ)^{N_{>s}} · A^loc_{s,i}**, where N_{>s} = Σ_{j>s} n_j is the number of trainable tokens generated after segment s.
   - Without this, naive per-segment GAE puts the shared terminal reward at the end of **every** segment, so early segments wrongly believe the outcome is near.
9. **No KL.**
10. **Results:**
    - GLM-4.5-Air: SWE-bench Verified **66.8% (+7.0)**, Terminal-Bench 2.0 24.5% (+3.1).
    - GLM-4.7-Flash: 56.0% (+5.5) and 20.2% (+6.8).
    - Compacted evaluation: 59.8 → 66.8.
    - The authors admit inconsistent gains under single-window evaluation.

**Other context management:**
- **MiniMax:** context management is an **explicit agent action** folded into the dynamics, s_{t+1} = f(s_t, a_t, o_t), so the policy anticipates it during RL. Paired with M2's (s_t, a_t) atomic unit.
- **Kimi K3:** context management is a **configurable harness module varied across task groups**, to avoid overfitting one mechanism.
- **Inference-time heuristics, not shown to be in RL:**
  - GLM-5 keep-recent-k (k = 5, BrowseComp 55.3 → 62.0) plus hierarchical HCM (beyond T = 32k, drop all tool history, 75.9);
  - LongCat hybrid (80K triggers a summary; max turns trigger discard-all; progressive discard thresholds; 55.8 → 73.1).
- **Thinking retention:**
  - GLM-5 Preserved Thinking;
  - Kimi K3's template supports preserved thinking only ("otherwise very unstable");
  - StepFun is selective (keep only the reasoning of the latest user instruction's tool chain), because dropping all thinking fails 100+ turn sessions and keeping all of it is too expensive;
  - DeepSeek-V4 interleaved thinking keeps all reasoning, across user turns.

### 6.4 Harness abstraction

- White-box vs black-box is the axis (Forge, Kimi Fig. 4).
  - **White-box:** the trainer sees the agent loop, the environment server and the reward server, and can manage context as an action.
  - **Black-box:** only the gateway sees tokens; the scaffold is opaque, including its context surgery.
- MiMo's Harness Pool runs several harnesses side by side (A/B/C).
- Kimi varies the harness and context mechanism per task group.
- Forge trains across 100k+ scaffolds; MiniMax argues for single-stage mixed-domain RL over specialist-then-merge (§5 route ④).

### 6.5 M12 requirements: middleware, multi-agent, compaction, harness

- **The buffer is a Data Pool** with Forge-style **tag/value separation** (MiMo TransferQueue):
  - the tag (small, indexed) = status ∈ {pending, running, finished, failure, masked}, group, task family/source, submit_version, versions touched, length, harness;
  - the value = the `crucible.rollout/v1` record.
  
  Scheduling decisions read tags only.
- **Gateway = the capture proxy** (M3/M10), with the same contract for white- and black-box harnesses. Black-box harnesses whose prompts are not prefix-extending (their own context surgery) produce `prefix_break` segments. They train only if the segmenting is recorded (§6.3), else they are masked; already the rule.
- **Async reward with a deadline.**
  - Verifiers and judges run off the trainer's critical path. `reward.deadline_s` is per tier.
  - A rollout whose *outcome* verifier misses the deadline stays `pending` (it is never trained as 0).
  - An *auxiliary* channel (rubric/GAR) that misses its deadline falls back to the outcome-only advantage, and the fallback is recorded (MiMo).
- **Compaction (CompactionRL recipe as `credit.segment_mode`):**
  - `trigger_remaining_tokens = 10240`, `keep_recent_steps = 2`. Crucible today: 80% of budget and k turns; make both configurable and record which was used.
  - Summary calls are trained with the episode advantage (exists).
  - Token-level normalization.
  - In critic mode, **cross-segment GAE** Â_{s,i} = (γλ)^{N_{>s}} · A^loc_{s,i}, with N_{>s} computed from `Segment` boundaries and trainable-token counts.
  - Group statistics on whole rollouts.
- **Multi-agent:**
  - **critical-path steps** computed from the World trace DAG: Σ over stages of main-agent steps + the slowest peer;
  - **frozen-subagent mode:** only the orchestrator's calls are trainable, and subagent turns get `loss_mask = 0`.
- **Harness diversity:** the sampler assigns harness@version (and compaction mode) per group, never mixing inside a group. The share per harness is logged per batch.

---

## 7. The author's six routes and the open problems (art. §5, §6, §7)

### 7.1 Six routes

| # | Route | Representatives | Systems implication for M12 |
|---|---|---|---|
| ① | sync / near-sync colocated + masking | Kimi K3 (colocated + partial + per-token regularization), Hunyuan (colocated sync GRPO + IcePop-style) | `colocated_sync` / `colocated_partial` + a token mask; smallest footprint; heavy memory juggling |
| ② | async + partial rollout + staleness control (orthogonal to colocation) | GLM-5 (disaggregated + version drop + DIS), DeepSeek-V4.1 (colocated + sample-level async + stale mask), LongCat (disaggregated, streaming multi-version), Ling 2.6 (colocated + partial + staleness manager) | the scheduler and staleness layer of §2.8; weight-sync speed decides how tight S can be |
| ③ | sequence-level IS (GSPO) instead of routing replay | Qwen, LongCat GSPO, Ling-2.6 agentic | `gspo` in the menu; no routing capture needed; coarse credit (multi-turn needs GSPO-token) |
| ④ | specialize, then distill / multi-teacher on-policy distillation | DeepSeek full-vocabulary OPD (V4 > 10 teachers, V4.1 > 40, replacing mixed RL), Kimi K3 MOPD (9 experts × reasoning strength, per-token dense reward in the RL framework), MiMo MOPD2 (fixed repetition for **$90k**), GLM cross-stage OPD (10+ experts in about 2 days), Qwen3-Coder-Next | teacher-serving pool + per-token teacher logprobs; a dense token reward into the same loss. The opposite position is MiniMax's single-stage mixed RL. |
| ⑤ | critic vs critic-free | critic: SAO, CompactionRL, MIS-PO, Seed; critic-free: GRPO/GSPO/CISPO | critic mode: value-model training is its own subsystem (below) |
| ⑥ | environment as infrastructure | DSec, AgentENV, Qwen Docker corpus, LongCat noise injection, Ling 220k images, StepFun environment-building RL | M1/M8/M10 (Crucible's core); M12 consumes them through the env pool |

**Critic-mode costs** that route ⑤ adopters each paid (§3.5.3.4–7):
- **SAO:**
  - TTUR, the critic updated **K = 2** times per policy update;
  - **frozen critic attention** (instability came from full-attention layers, not MoE);
  - scaled value pretraining;
  - length-adaptive GAE λ_policy = 1 − 1/(αl) with α = 1.5 and λ_critic = 1 (from VAPO);
  - **skip-observation GAE** Â(a_{i,N}) = δ + γλ Â(a_{i+1,0}), δ = r_t + γV(a_{i+1,0}) − V(a_{i,N}). Observation tokens are removed from the Bellman chain; token-level beats step-level (89.8/66.8 vs 85.8/60.5 mean and 87.3/62.8 last token).
  - Vanilla GRPO collapses at about 160 steps, VAPO at about 90; SAO is stable for about 1,000.
- **MIS-PO:**
  - γ = λ = 1;
  - Muon optimizer, weight decay 0.1;
  - actor lr 2e−6 / 20 warmup, critic lr 5e−6 / 50 warmup;
  - **4 actor : 12 critic mini-batches** per round;
  - unbiased KL 0.001 at the end;
  - truncation value bootstrap.
- **Seed:**
  - shared critic initialized from the RM, 100-step warmup;
  - per-task KL (general 1e−5, verifiable 0);
  - value pretraining + decoupled GAE.

### 7.2 Open problems (art. §6) and what M12 does about each

1. **Colocated vs disaggregated: no conclusion.** MiMo, Qwen, StepFun and Seed did not even state theirs. → M12 makes topology a config and logs bubble time per phase (§1.3).
2. **No standard staleness cap** (MiMo 4, StepFun about 20, the rest unpublished). → Staleness is a knob with three enforcement layers; logs spans, staleness and evictions (§2.8). A sweep over S is a natural M12 experiment.
3. **Eliminate the mismatch at the system level, or correct it algorithmically?** The sharpest split. Ring-1T admits IcePop is not perfect. → M12 measures (`rollout_probs_diff`, r^dis) before correcting, ships both FP32-head/QDQ hooks and the loss menu, and can report the gap.
4. **Routing replay or not?** (Qwen: no; MiMo R3; DeepSeek spliced.) → Measure Σ_k; R3 is optional and needs `Turn.routing`.
5. **The value of a critic.** → Critic mode exists for single-feedback enterprise workflows (DESIGN §10.3.5), with the SAO/MIS-PO engineering list as its spec.
6. **Is process reward worth it?** Seed: no. Ling and MiniMax: yes. MiMo, Kimi and LongCat: no independent PRM. → Crucible's dense credit is potential-based and outcome-anchored (M6, long-horizon Φ). M12 makes its use a switch (`credit_unit`), so the question becomes testable.
7. **The reward-hacking arms race.** MiMo ToolRep: penalties inside RL generalize poorly and slowly; rerunning 20 steps of MixRL cost about **$2.31M**. → In-loop gateway interception; a pass-rate-jump alarm (DESIGN §10.3.8); confirmed-hack rate.
8. **Environment noise and reproducibility.** → Infra-mask vs agent-caused failure; group backfill policy; flaky admission already stricter in M9.
9. **Long-horizon credit assignment is open.** Cross-trajectory GAE, skip-observation GAE, GAR and DAG critical path are all approximations without bias bounds. → M12 implements cross-segment GAE + skip-observation masking in critic mode and keeps Φ-telescoping as Crucible's own bias-free shaping (DESIGN §10.4).

**The author's closing thesis** (§7): the field is choosing between pushing the system toward clean on-policy training (bitwise alignment, strict staleness, routing replay) and **"embracing imperfection, with algorithmic robustness as the backstop"** (sequence IS, masks, binary acceptance). The latter is winning, but the two will coexist. The next winners will crack **long-horizon credit assignment** and **reproducible, noise-robust, large-scale environment synthesis**, and will close the model–harness–environment co-design loop.

---

## 8. M12 requirements checklist (numbered, testable)

Each item is phrased so that a unit, integration or simulation test can decide it. "Sim" means a deterministic discrete-event simulator of rollout durations, like the article's TSX demo, run with no GPU.

### Interfaces and versions

1. M12 exposes four interfaces mapped to the article's planes: `Scheduler` (control), `DataPool` (data), `StateStore` (state: World checkpoint + policy `state_dict` + paused-call registry) and `WeightPublisher` (weight). Every message across them carries an integer `policy_version`.
2. `WeightPublisher.publish()` increments `policy_version` by exactly 1. Every engine response carries the version that served it, and the capture proxy writes that value, not a wall-clock inference, into `Turn.policy_version`. Test: a publish in the middle of a 100-call rollout gives calls tagged v and v+1, with no call tagged by a version that was not serving when the call started.
3. With `version_switch = "call"` (the default), no `Turn` has completion tokens produced by two versions. Test: inject a publish while a call is decoding. The call is either finished on the old version (disaggregated) or aborted and re-issued whole (colocated). It never stitches.
4. A completion that crosses a version boundary is accepted only when `Turn.version_spans` is present and covers every completion token. Otherwise the rollout is rejected by `valid_for_training()`.
5. Publish latency (`weights/publish_s`) and re-prefill tokens (`rollout/reprefill_tokens`) are logged every step.

### Topology and placement

6. `topology ∈ {colocated_sync, colocated_partial, disaggregated_stream}`, with `colocated_partial` the default. Switching topology changes no loss, credit or record code: the same rollout records produce identical loss values under every topology (golden test).
7. In colocated modes, `offload.train_state_on_rollout ∈ {none, cpu, nvme}` and `offload.kv_on_train ∈ {discard, cpu_pool}` are honoured. Test: GPU memory held by the trainer during rollout is ≤ 5% of its training peak when `cpu` is set.
8. Environment workers are a separate pool (`env.workers`). Test: with the GPU engine stopped, environment steps already issued still complete. With environments blocked, the engine serves other rollouts' calls.
9. Bubble time per phase (`idle/rollout_s`, `idle/train_s`, `idle/env_wait_s`) is logged every step.

### Async control flow and synchronization points

10. `colocated_sync` reproduces on-policy training: every trained rollout has staleness 0 and `trajectory_spans` = 1.
11. `colocated_partial` at the sample barrier: (a) aborts every in-flight model call; (b) releases engine KV when `discard`; (c) leaves every World's state intact; (d) re-issues the aborted calls under the new version after publish. Test: World checkpoints before and after a barrier are byte-identical, and no environment action is replayed.
12. `disaggregated_stream` never pauses generation for training. Test (sim with a train step of 4 ticks): engine utilization stays > 90% while the trainer trains.
13. `publish_every_k_updates = K` publishes exactly every K optimizer steps. `reset_optimizer_on_publish = true` zeroes optimizer moments on each publish (GLM).
14. Async modes run `warmup_batches` (default 1) of generation before the first optimizer step.

### Granularity and admission

15. The dispatch unit is one rollout. Admission per source follows the DeepSeek rule: a group of n is admitted only when `free_slots(source) ≥ n`, where `inflight_cap(source)` is configurable. Test (sim): the number in flight per source never exceeds its cap and never drops below cap − n + 1 while the source has prompts left.
16. GRPO advantages are computed only for terminal groups. Critic mode (n = 1) has no group barrier. Test: in critic mode a finished rollout becomes sampleable without waiting for any other.
17. Infra-masked group members are handled by `group_backfill ∈ {pad, refill_member, drop_group}`, and the choice is recorded per group.
18. `loss_agg ∈ {token_mean_global, seq_mean_token_mean, prompt_mean, seq_ratio, const_max_len}`, default `token_mean_global`. Golden test, the DAPO example: G = 2, |o₁| = 100, |o₂| = 10,000. Per-token weights are 5e−3 / 5e−5 under `seq_mean_token_mean` and 9.9e−5 under `token_mean_global`.
19. Sequence lengths are balanced across data-parallel ranks before each update. Test: the max/min summed tokens per rank ≤ 1.1 on a skewed batch.

### Staleness

20. `staleness(r) = v_train − r.oldest_version()`. With `staleness_strategy = drop` and `max_staleness = τ`, no trained rollout has staleness > τ (assertion, as in the MiMo `drop` path), and every eviction refills exactly one group per evicted group.
21. With `staleness_strategy = wait` (the default), sampling blocks while any pending or running rollout has staleness ≥ τ, and no rollout is ever dropped for staleness. Test (sim, 20% of rollouts 10× longer): zero staleness evictions and max trained staleness ≤ τ.
22. Admission gating: in-flight rollouts ≤ S × B (S = `max_staleness`, B = groups per optimizer step). Test (sim): at every tick |in-flight| ≤ S × B.
23. The GLM worked example is a unit test: τ = 3, v′ = 10. Versions (6,7,8) are dropped; (7,8,9) are kept.
24. Logged every step: `trajectory_spans/{mean,max}`, `staleness/{mean,max}`, `staleness_worst/{mean,max}`, `evicted_staleness/{mean,max}`, `admission_blocked_s`, `sample_wait_s` and the share of on-policy samples.
25. Optional `max_offpolicy_frac`: when set, no batch has more than that share of samples with staleness ≥ 1.

### Windowed FIFO and length bias

26. `buffer_policy = windowed_fifo` takes only terminal items whose submission index < head + W, where head = the oldest untrained submission and W = ⌈`fifo_window_frac` × N⌉ (default 0.3). Inside the window it picks earliest-finished first.
27. The windowed-FIFO simulator reproduces Fig. 9:
    - W = 1 equals strict FIFO (trainer idle while T0 runs);
    - W = N equals greedy;
    - with W = 4, finished T4, T5, T7 are not taken while T0 is untrained.
28. Windowed-FIFO bound: in the sim with batch = 8 and W = 4, max off-policy lag ≤ 10 (the article's bound 8 + 3 − 1).
29. Length-bias monitor: each step logs the KS distance between batch response lengths and the steady-state length distribution (a trailing window), plus the share of hard (long) tasks in the first 10 batches. With `drop_early_short` enabled, returns below the configured percentile during the first M steps are dropped and counted.
30. `inflight_cap` per data source is enforced (DeepSeek source-mix control). Test: the per-source share of a batch stays within ±10% of the configured mix in the sim.

### Mismatch

31. `rollout_probs_diff`, quantiles of r^dis (p1, p50, p99), the masked-token share and the clipped-token share are logged every step under every correction, including `none`.
32. `mismatch.correction` implements tis, icepop, kpop, dis, mimo4, mispo, hispo, cispo and gspo as pure functions, with the article's formulas as docstrings. Default: `dis(ε_ℓ = 0.8, ε_h = 3.0)`.
33. Golden tests:
    - KPop: p = 5e−5, q = 1e−5 gives D^B = 4.05e−5 ± 1e−7; p = 0.9, q = 0.45 gives 0.4534 ± 1e−4. φ = 0.01 keeps the first and masks the second.
    - MIS-PO: a trajectory with geometric-mean ratio 0.995 is rejected, 0.999 accepted; a token ratio of 2.1 is masked.
    - IcePop (α = 0.5, β = 5): r^dis = 6 gives weight 0; r^dis = 0.7 gives weight 0.7 × r^stale.
    - DIS: weight 0 outside (1 − ε_ℓ, 1 + ε_h) regardless of the sign of Â.
    - HisPO: a negative-advantage token with r = 50 contributes a bounded value (ε^neg_high clip).
    - CISPO: the gradient equals r̂ Â ∇log π (stop-gradient verified by autograd).
34. Corrections that need π_θold (icepop, hispo, cispo, gspo, mispo) refuse to run when a rollout spans more than one version, unless an explicit `old_logprob_source ∈ {recompute_current, rollout}` is set. DIS and mimo4 run on rollout logprobs alone.
35. `lm_head_fp32` defaults to true. The run record states the LM-head precision of both the training engine and the inference engine.
36. For MoE policies, Routing Confidence Σ_k is logged. `routing_replay = r3` requires `Turn.routing` for 100% of completion tokens and is otherwise refused.
37. Rollout `top_p` defaults to 1.0. If top_p < 1 and the correction uses π_rollout, `Turn.candidates` must be present, and training logprobs are renormalized over it. Test: renormalized train logprobs on a toy vocabulary match the engine's within 1e−5.
38. An optional `post_update_transform` (e.g. QDQ to the rollout quantization grid) runs before every publish. With QDQ on, the trainer's expert weights after the transform are bit-identical to what the engine loads.

### Middleware, reward, harness

39. The DataPool stores tags (status, group, source, submit_version, versions touched, length, harness) separately from values (`crucible.rollout/v1`). Scheduler decisions read tags only. Test: the scheduler runs with values stored remotely and never fetches them.
40. Reward is asynchronous with deadlines. A rollout whose outcome verifier misses `reward.deadline_s` stays pending (never trained as 0). An auxiliary channel (rubric/GAR) that misses its deadline falls back to the outcome-only advantage, and `credit_fallback = true` is recorded.
41. A detected-and-blocked shortcut probe returns refusal or dummy content, and the episode continues (GLM-5.2). Only an *exposed* shortcut masks the episode (P27). Test: a blocked probe produces an unmasked, trainable rollout.
42. An infra failure gives `mask_sample = true` and `reward = None`. An agent-caused environment break gives `failure_kind = agent_broke_env` and reward 0, and is never masked.
43. Harness@version and compaction mode are assigned per group, never mixed inside a group. The per-batch harness share is logged.

### Long horizon, compaction, critic, multi-agent

44. Compaction parameters (`trigger_remaining_tokens`, default 10240 per CompactionRL, or a budget fraction; `keep_recent_steps`, default 2) are configurable and recorded in the run record.
45. Summary (compaction) calls carry the episode's advantage and are trainable. No separate summary-quality reward exists by default.
46. In critic mode, cross-segment GAE is implemented: Â_{s,i} = (γλ)^{N_{>s}} · A^loc_{s,i}, with N_{>s} the number of trainable tokens after segment s. Golden test: for a 3-segment rollout with known values, the advantages match a hand computation. With S = 1 it reduces to standard GAE.
47. Critic mode masks observation tokens out of the Bellman chain (skip-observation GAE). Test: the advantages are unchanged when observation token contents are permuted.
48. Critic mode exposes `critic_updates_per_policy_update` (default 2, SAO TTUR), `freeze_critic_attention` (default true), λ_policy = 1 − 1/(αl) with α = 1.5 and λ_critic = 1, and truncation bootstrap (R̂ = V(s_T) for truncated rollouts).
49. Multi-agent worlds report `critical_steps = Σ_stage (main_steps + max_i sub_steps_i)` from the trace DAG. `frozen_subagents = true` sets `loss_mask = 0` on every non-orchestrator call.
50. Prefix sharing (optional): when enabled, the loss on a GRPO group computed with shared-prefix packing equals the unpacked loss to 1e−6 (Forge's "zero impact on loss").

### Monitors (silent-failure detection)

51. Every step logs:
    - staleness (24) and mismatch (31) metrics;
    - truncation share;
    - mean and p99 response length;
    - entropy;
    - the share of zero-variance groups;
    - confirmed-hack rate.
52. A training pass-rate jump of more than `alarm.delta` within `alarm.window` steps raises an alarm and triggers a full-tier re-audit of the batch's rollouts (DESIGN §10.3.8; LongCat's step-80 Lean4 jump of 97.9% → 27.9% after the fix is the motivating case).

---

## Appendix A. Facts I could not pin down (do not cite as fact)

- **Forge's "four interfaces":** asserted in art. §3.1.4 but not enumerated in the article or the Forge blog.
- **LongCat "new requests only go to the old version (load 63%)":** in the Table only. Unexplained in the article and absent from the Meituan blog. My version-group reading in §2.4 is an interpretation.
- **DORA speedup:** the article says 2–4× vs sync; the Meituan blog says 3×.
- **Fig. 4 attribution to Kimi K2.5:** inferred from its position and content; the caption has no lab name.
- **DeepSeek "token-level WAL":** in the Table only. The prose says only "pause at any token boundary, reuse persisted state, no re-prefill".
- **MiMo's placement:** colocated is the article's inference from its control flow and the open recipe; the report does not say.
- **The article does not use "R2"**; see §5.3, group 4.
