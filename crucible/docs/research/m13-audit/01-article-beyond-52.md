# 01: The article beyond the 52 trainer items (requirements and gaps)

**Source:** Wei Ziyang, *Agentic RL Infra* (2026), `RL-Environments/07-articles/qingkeai-2026-agentic-rl-infra.md`. Section numbers are the article's.
**Baseline:** the 52-item checklist (`docs/research/m11-m12/06-article-systems.md` §8; status in `docs/modules/M12-train.md` §9).
**Code audited:** `src/crucible/` at `90200ec`, plus the critic commits `9cc6db3` / `e03032a`, on 2026-10-10. An uncommitted `src/crucible/train/state.py` (a run-level StateStore) appeared during the audit. It is noted under A2-1 but not counted as done.

**Scope.** The 52 items cover the scheduler, staleness, mismatch corrections, the trainer and the weight plane. This audit covers what the article asks of everything else. An item the 52 already cover appears here only when this audit adds to it.

**Legend.**
- **Status:** done, partial or missing, with the `file:symbol` checked.
- **Priority:** P0 = needed to claim support; P1 = expected of a serious framework; P2 = specialised.
- **Effort:** S < 2 days, M ≈ 1 week, L > 2 weeks.

---

## 1. State plane (art. §1, §3.2.3.1, §3.2.4, §3.8.3)

The article's state plane saves and restores *KV cache, environment state, tool sessions and partial rollouts*.

Crucible has episode-level state: `World.snapshot/restore`, and `runtime/checkpoint.Checkpointer` for M10 retries. At `90200ec` the training system itself had no persistent state:
- `train/pool.DataPool` and `train/scheduler.Scheduler` have no `state_dict`;
- `TorchTrainer.state_dict` returns weights only;
- `Runtime._run` cancels every in-flight task when it ends.

| ID | Requirement | Art. | Status / code | Missing | Acceptance | P | E |
|---|---|---|---|---|---|---|---|
| A2-1 | Run-level checkpoint and resume: trainer weights and optimizer, version, pool tags and values, scheduler counters, source cursors, RNG | §1 control plane "fault recovery"; §3.2.4 Ling | **missing** at `90200ec`; in progress in the uncommitted `train/state.py` | committed StateStore; resume test | Kill at step k and resume on `ScriptedEngine`: identical selections, versions and metrics; nothing trained twice; no prompt lost | P0 | L |
| A2-2 | Paused-rollout registry: interrupted rollouts persist World checkpoint, policy state and partial call; a restart resumes them | §3.2.3.1 Kimi | **partial**: `runtime/checkpoint.Checkpointer` (M10 only); `GenRequest.resume_*` in memory; the `state.py` draft restarts in-flight rollouts from scratch | World checkpoints in the train runtime | Crash during a barrier and resume: running rollouts continue with `resumed_from`, fingerprints unchanged | P1 | M |
| A2-3 | Resumed rollouts are dispatched before new admissions | §3.2.3.1 Kimi | **partial**: priority exists in `train/sim.py` only; `Runtime._call` re-issues without priority | engine queue priority | Re-issued calls precede the first call of any new rollout | P2 | S |
| A2-4 | Resume at a token boundary without re-prefill: KV survives the weight switch, and its version is recorded | §3.2.3.1 DeepSeek-V4.1; Table "token-level WAL" | **partial**: `version_switch="token"` keeps tokens and spans but re-prefills (`reprefill_tokens`); `VLLMServerEngine.kv_modes=("discard",)`; no engine offers `cpu_pool` | a KV-retaining engine; `Turn.kv_spans` | 0 prefill tokens for a resumed call; the record names the KV version | P1 | L |
| A2-5 | Sandbox pause while waiting for inference, releasing resources | §3.8.3 Kimi (paused for up to 98% of a sandbox's life) | **missing**: no pause in `world/backends/base.Backend` or `DockerSandbox`; Worlds stay live through barriers | `Backend.pause/resume`; `env/paused_share` | Fingerprint unchanged across a pause; container CPU ≈ 0 while paused | P1 | M |
| A2-6 | Fork: a second live World from a `WorldRef`, for judging without side effects | §3.8.3 Kimi fork | **partial**: snapshot/restore; `credit/operators/branch_value` restores in place; separate verifier image (`verify/stage.py`) | `World.fork(ref)`; latency metric | Grading the fork leaves the original fingerprint unchanged; p50/p99 fork time logged | P1 | M |
| A2-7 | Snapshot and restore latency and size per adapter; incremental snapshots | §3.8.3 Kimi (133 ms / 49 ms) | **missing**: `world/state/fs.py` and `sql.py` take full snapshots; no timing | incremental fs; benchmark | p50/p99 ms and bytes reported; after a one-file edit, an incremental snapshot ≤ 10% of a full one | P2 | M |
| A2-8 | Live tool sessions (shell, browser, MCP) are snapshotted or refused by name | §1 "tool sessions" | **partial**: `Backend.snapshot_state`; `quiesce` kills leftover processes | fail-closed detection | Snapshot with an open session: the session is restored, or a named error is raised | P1 | S |

## 2. Route ④: on-policy distillation and multi-teacher OPD (art. §5, Table)

Nothing exists: no teacher, no distillation, no reference model, no KL. Yet:
- DeepSeek-V4.1 uses OPD (> 40 teachers) instead of mixed RL;
- Kimi K3 MOPD puts a per-token dense reward into the RL loop;
- GLM-5.2 merges 10+ experts in about 2 days.

One hook exists: in critic mode `TorchTrainer._items` already takes per-token advantages, which is where a dense token reward fits.

| ID | Requirement | Art. | Status / code | Missing | Acceptance | P | E |
|---|---|---|---|---|---|---|---|
| A2-9 | Teacher pool: N teacher engines; teacher logprobs for finished rollouts, off the critical path, with deadlines | route ④; Kimi MOPD (9 experts × effort) | **missing**: `train/engine.EnginePool` serves the policy only | teacher pools; `Turn.teacher_logprobs`, `teacher_id` | 100% token coverage; a missed deadline masks or records a fallback; teacher time excluded from engine bubbles | P0 | L |
| A2-10 | OPD objective: per-token reverse KL (log π_T − log π_θ) or full-vocabulary KL, alone or mixed with the RL advantage | route ④ DeepSeek full-vocab OPD | **missing** (`train/objectives.py`, `train/mismatch.py`) | loss term; mixing weight | Teacher = policy gives a 0 gradient; full-vocab KL matches a hand computation within 1e-6; weight 0 reproduces the RL step bitwise | P0 | M |
| A2-11 | Teacher routing per sample, with reasoning-effort conditioning | route ④ Kimi, MiMo MOPD2 | **missing** | routing map; `opd/kl/<teacher>` | Each record's `teacher_id` equals `map[source]` | P1 | M |
| A2-12 | Teacher/student tokenizer compatibility gate | route ④ (implied) | **missing** | check | Mismatched vocabularies refused before any rollout | P1 | S |
| A2-13 | Forward-only models kept on CPU and materialised on demand | §3.1.3 Kimi | **missing**: `TorchTrainer.offload` covers the policy only | teacher offload | Teacher resident memory ≤ 5% of its weights during rollout | P2 | M |
| A2-14 | KL to a reference, coefficient per source | §3.5.3.5 StepFun (0.001); §3.5.3.7 Seed (1e-5 / 0); Table Seed "small KL against hacking" | **missing**: no KL in `TrainerConfig` | k3 estimator; reference model | Golden k3; KL = 0 when policy = reference; per-source map honoured | P1 | M |

## 3. Reward-engineering pipeline (art. §3.6, Table)

The article's chain is *verifier → rubric/GRM → hack zeroing → group redistribution → shaping → GAE or broadcast* (§3.6.4). Crucible has:
- the verifier (M7);
- validated binary rubric items (`credit/operators/rubric_item.py`, `verify/judge_item.py`);
- the gated composition (`credit/compose.compose`);
- async per-record grading (`Runtime._grade`: `reward_fn(rec)`, `aux_fn(rec)`).

**Structural gap: every grading hook sees one record.** GAR, tournaments, pairwise GenRM and "zero the hack, then recompute the group statistics" all need the whole group. The group statistics exist only inside `TorchTrainer._items`.

| ID | Requirement | Art. | Status / code | Missing | Acceptance | P | E |
|---|---|---|---|---|---|---|---|
| A2-15 | Group grading stage: `group_fn(members)` after the group is terminal and before advantages; rewrites rewards and advantages; async, with a deadline and a fallback | §3.6.3.2 MiMo GAR; Kimi tournament | **missing**: `Runtime._grade` grades per record | the stage; group-level `grading` status | Sees all n members at once; when late, sets `group_credit_fallback` and training uses the raw advantages | P0 | M |
| A2-16 | MiMo GAR: f_i ∈ (0,1] on passing members; mass-conserving Σ_P A′ = Σ_P A; λ cap; zero hacks, then recompute the mean; final centring | §3.6.3.2c | **missing**: planned in `ARCHITECTURE.md` M12 `credit.py` and DESIGN App. A | operator | Hand-computed golden case; passing mass preserved to 1e-9; failing members unchanged before centring; all-pass and all-fail groups untouched | P0 | S |
| A2-17 | MiMo GRS: R = R_test·S_sol·S_beh, with solution and behaviour rubrics an agent writes offline, grounded in the task rather than the rollouts | §3.6.3.2b | **partial**: additive α·rubric with `compose.MAX_ALPHA = 0.1` caps the spread among passing solutions at 10%; no rubric-writing agent | product mode; builder | Failures score 0; all-pass group variance = Var(S_sol·S_beh); items citing a rollout id rejected | P1 | M |
| A2-18 | Tournament and pairwise Bradley–Terry judging: O(n) against a fixed reference (StepFun); judge protocol read → rubric → score → scorepad (Kimi) | §3.6.3.2d, f | **missing**: `credit/judge.py` judges one trajectory | BT fit; protocol | BT matches a synthetic win matrix; exactly n calls per group; both orders judged, agreement logged | P1 | M |
| A2-19 | Verbosity budget: a candidate longer than σ·ℓ₀ loses automatically | §3.6.3.2d Kimi | **missing** | ℓ₀ per task | Length σ·ℓ₀ + 1 loses every comparison | P2 | S |
| A2-20 | GenRM extras: length penalty on confidence; zeroing for fabricated citations, overconfidence, language mixing; MetaRM; ternary verdict → asymmetric binary | §3.6.3.2f StepFun | **missing**: binary items only | rules | Each rule fires on its fixture | P2 | M |
| A2-21 | MiniMax composite α·r_process + β·r_speed + r_perf, with r_speed = h(T/T_base) and the reward-to-go advantage | §3.6.3.1 | **partial**: `compose` efficiency (β ≤ 0.05) is solution performance, not agent wall time; `turn_advantages(mode="baseline")` is a shaping baseline | speed term; reward-to-go | h is monotone decreasing; the terminal r_perf appears in every step's advantage | P1 | M |
| A2-22 | Ling focus reward: per-dimension on-policy saturation shifts weight to unsaturated dimensions | §3.6.3.2g | **missing**: validity is offline AUC (`credit/validity.py`) | monitor | A dimension at ≥ 0.95 pass share loses weight; weights logged | P2 | S |
| A2-23 | GLM mix of rule, ORM and GRM channels per source, with human style anchors | §3.6.3.2e | **partial**: rule gates and LLM items; no ORM; no anchors | channel registry | Anchor present in the GRM prompt; channels switchable per source | P2 | M |
| A2-24 | Penalty for an incomplete trajectory (turn limit hit) | §3.6.3.1 Qwen | **partial**: `runtime/infra.LimitPolicy` is score or mask only | `penalty:<c>` | `on_steps="penalty:-0.5"` gives −0.5 | P1 | S |
| A2-25 | Tool-format penalty on the offending call's tokens only | §3.6.3.1 Qwen; MiniMax r_process | **partial**: `World._call_tool` labels `invalid_arguments` / `unknown_tool` as policy faults; `Turn.advantage` is per call; nothing links them | mapping | One malformed call: only its tokens get −c | P1 | S |
| A2-26 | Grader compute accounted as a scaling axis | §3.6.1 MiMo (12.7% of cost) | **partial**: deadlines and late counts only | metrics; concurrency cap | `grader/s`, `grader/usd`, `grader/share` per step | P1 | S |
| A2-27 | The actor serves as the GRM, and the GRM is RL-trained | §3.6.3.2a DeepSeek-V4 | **missing** | judge routing to the policy pool | Judge calls tagged `kind="judge"`, trainable when enabled | P2 | L |

## 4. Reward-hacking layers (art. §3.7)

Layers 1 and 3 are strong:
- **Environment hardening (layer 1):** `world/guard` (policy user, egress probe, `git_scrub` / `git_trust`), `world/leaks`, `verify/leak_audit`.
- **Verifier hardening (layer 3):** `verify/runtime/patterns/static.py`, `perf.py`, A13.

Layers 2 and 4 are weak:
- Probe detection runs after the fact: `connectors/agent_events.annotate_probes` reads the finished trace, and `actors/policy.run_episode` masks on exposure.
- `Runtime._train` reads `reward_components["hack/confirmed"]`, but nothing in `src/` writes it.
- `verify/tiers.audit_sample` is never called from `train/runtime.py`.

| ID | Requirement | Art. | Status / code | Missing | Acceptance | P | E |
|---|---|---|---|---|---|---|---|
| A2-28 | Online interception before execution at `World._call_tool`: refusal or dummy content, rollout continues | §3.7.3.5 GLM-5.2 | **partial**: only environment blocking and after-the-fact probes | interception; `hack/intercepted` | `git clone github.com/x/y` not executed (fingerprint unchanged), returns refusal, episode trainable; closes checklist #41 | P0 | S |
| A2-29 | Two-level detector: rule recall, then an async LLM judge for precision; produces `hack/confirmed` and a confirmed-rate alarm | §3.7.2; §3.7.3.6 MiMo < 2% | **missing**: consumer only | judge stage; producer | A confirmed hack zeroes the reward via A2-15; an unparseable judge never confirms; alarm above threshold | P0 | M |
| A2-30 | Task-aware git back-link rule: the task's `github.com/{repo}` together with a network keyword | §3.7.3.1 Qwen | **partial**: generic `PROBES` (`code_host`, `git_history`); guard `ANSWER_SOURCES` | repo slug; online | `git remote add origin https://github.com/{repo}` intercepted | P1 | S |
| A2-31 | Full-tier audit of sampled training rollouts; fast-pass / full-fail = hack signal | §3.7.4; DESIGN §10.2 | **partial**: `verify/tiers.audit_sample` unused | wiring | About r re-graded (deterministically); `hack/fast_full_disagree` logged | P0 | S |
| A2-32 | Hack-agent red-team loop until no exploit remains | §3.7.3.6 MiMo | **partial**: static A13 (`accept/checks/extended.a13_attacks`), A6, toy forge Adversary | adaptive agent loop | Planted leak found in round 1, 0 after patch; card records `redteam_rounds` | P1 | L |
| A2-33 | AST validator plugins per language, one fixture per pattern | §3.7.3.3 LongCat Lean4 (9 patterns, 97.9% → 27.9%) | **partial**: regex rules for C and Python in `static.py`; no AST; no Lean | registry; parsers | A Lean4 validator catches all 9 patterns | P2 | M |
| A2-34 | Soft-hack detection: compatibility branches, broad exports, swallowed exceptions, relaxed validation | §3.6.3.2, §3.7.3.7 | **missing**: survey action §6.3 not built | mutants; scope rubric | Each fixture scores below the minimal patch | P1 | M |
| A2-35 | seccomp/AppArmor profile and eBPF-style network policy per sandbox | §3.7.3.2 DeepSeek | **partial**: `IsolationSpec` (uid, pids, memory, egress) | syscall profiles | `mount`, `ptrace`, raw sockets denied (red-team test) | P1 | M |

## 5. Environment noise (art. §3.8.3)

| ID | Requirement | Art. | Status / code | Missing | Acceptance | P | E |
|---|---|---|---|---|---|---|---|
| A2-36 | Admission requires 8 consistent reruns; drop instances whose gold patch is unstable | MiMo, Ling | **done**: `accept/checks/core.a4_flake` (`a4_stable_vectors = 8`, 40 draws) | — | existing | P0 | — |
| A2-37 | Infrastructure failures masked, never scored | MiMo Penalty Module | **done**: `runtime/infra`, `credit/operators/infra_mask`, `World._infra`, breaker | — | existing | P0 | — |
| A2-38 | Flakiness checked during training: re-grade a sample K times; mask flipping verdicts; quarantine the task | §3.8.1; §6 | **missing**: A4 runs before training only | — | Injected flaky grader: flippers masked, task quarantined, `grader/flip_rate` logged | P1 | M |
| A2-39 | Agent-caused crash = failure plus a recoil signal below failure | DeepSeek | **partial**: `crucible:agent_broke_env` scores 0 (`core/schema/run.py`) | negative option | `broke_env_penalty = -c` gives reward < 0, never masked | P2 | S |
| A2-40 | Noise-injection curriculum (instruction and tool noise, solvable), advancing on the clean–noisy gap | LongCat (τ²-Noise 62.2 → 67.1) | **missing**: `runtime/inject.py` injects infra faults for M10, not noise the policy sees | noise layer at `_call_tool`; schedule | Oracle passes at every level; level advances when gap < δ | P1 | M |

## 6. Sandboxes and environment synthesis (art. §3.1.4, §3.8.3, route ⑥)

| ID | Requirement | Art. | Status / code | Missing | Acceptance | P | E |
|---|---|---|---|---|---|---|---|
| A2-41 | Sandbox tiers: function call, container, microVM, full VM | DeepSeek DSec | **partial**: `BackendCapabilities.isolation`; local, Docker (`world/sandbox.DockerSandbox`), Harbor, agentenv (Modal, E2B) | microVM / VM | A microVM backend passes the M1 red-team suite and the snapshot contract | P1 | L |
| A2-42 | Startup p50/p99, create QPS, failures and concurrency; warm pool | Ling ASandbox (100 ms, 5000 QPS) | **missing**: `train/envpool.EnvWorkerPool` counts wait and busy time only | `env/start_s`; warm pool | Warm-pool p50 below the SLO; 10k concurrent function-call Worlds keep loop lag < 50 ms | P1 | M |
| A2-43 | Multi-host placement with capacity and admission backpressure | DSec (2500+ containers per node) | **missing**: in-process threads; `runtime/run.py` semaphore | placement engine | Capacity never exceeded; full hosts block admission (`env/placement_blocked_s`) | P2 | L |
| A2-44 | PR-to-environment pipeline: image build, gold and F2P stability, QA-agent and validity denoising, yield per stage | Qwen 807,693; GLM 10k+; Ling 220k; LongCat tool graphs | **partial**: `forge/` (DAG, dedup, learnability), M9 as validity filter, importers (MiMo 266, AWM 100, OpenEnv, Harbor, NeMo-Gym) | PR mining; build farm | 50 PRs → accepted tasks, yields reported and reproducible | P1 | L |
| A2-45 | Environment building as a trainable task family | StepFun (≈ 40% build success) | **partial**: M8 builder role (toy) | task type; reward | Graded by A1/A2/A4; builder success rate is a forge metric | P2 | M |

## 7. Context management (art. §3.8.3, §3.8.4)

`actors/policy.ContextConfig` offers `compact | none`: a policy-written summary at trigger × budget, keeping the last 3 turns.

| ID | Requirement | Art. | Status / code | Missing | Acceptance | P | E |
|---|---|---|---|---|---|---|---|
| A2-46 | Strategy registry: keep-recent-k (GLM k = 5), discard-all, tiered HCM (T = 32k), LongCat hybrid (summary at 80k + discard-all + progressive thresholds) | §3.8.3 | **partial**: summary only (`LLMToolPolicy._maybe_compact`) | strategies; `strategy` on each segment | Each strategy is tested on a scripted transcript and recorded in `segments` | P1 | M |
| A2-47 | Context management as a policy action (a `context.compact` / `context.drop` tool) | §3.8.3 MiniMax | **missing**: compaction is triggered by the harness | tool; `mode="action"` | A policy call yields a trainable segment; no automatic trigger in this mode | P1 | M |
| A2-48 | Thinking history: preserved / selective (StepFun) / interleaved (DeepSeek), matching the deployment template | §3.8.4 | **partial**: reasoning replayed verbatim; `train/chat.py` has no thinking handling, so exact TITO keeps `<think>` where a deployed template (Qwen3) strips it | modes; template check | TITO holds in each mode (or a new segment is recorded); a template mismatch is refused | P1 | M |
| A2-49 | Swarm context sharding: subagents with bounded contexts return only their output | §3.8.4 Kimi K2.5 | **partial**: multi-actor Worlds (P28) | `spawn` with a budget | Only the final message reaches the orchestrator; tokens per actor recorded | P2 | M |

## 8. Multi-agent (art. §3.6.3.4)

| ID | Requirement | Art. | Status / code | Missing | Acceptance | P | E |
|---|---|---|---|---|---|---|---|
| A2-50 | Frozen PARL subagents end to end: a frozen engine that never gets publishes; no gradient from its turns | Kimi K2.5 | **partial**: `train/credit.token_credit(frozen_actors=)`, `bridges/common.flatten`; not in `TrainerConfig`; one-policy `EnginePool` | wiring; frozen engine | Subagent engine stays at v0; autograd shows a 0 gradient | P1 | M |
| A2-51 | r_parallel + r_finish + r_perf with λ₁, λ₂ annealed to 0 (a shared schedule utility, reused by A2-14 and A2-40) | Kimi PARL | **missing** | schedules | After `T_anneal` reward = r_perf exactly | P2 | S |
| A2-52 | DAG critical-path latency penalty built from spawn and wait dependencies | DeepSeek Agent Team; Kimi critical steps | **missing**: no `critical_steps` in `src/` (checklist #49 counts steps only) | DAG; wall-time variant | An off-path parallel subtask leaves the penalty unchanged; a longer critical branch raises it | P1 | M |

## 9. Multi-domain, one hybrid run (art. §0, §3.2.4, Table)

| ID | Requirement | Art. | Status / code | Missing | Acceptance | P | E |
|---|---|---|---|---|---|---|---|
| A2-53 | Per-source rollout function or harness, reward function and loss settings in one run | §0 MiMo "You Only RL Once" | **partial**: per-source `inflight_cap` / `mix` **done** (`SchedulerConfig`, checklist #15); `Runtime.__init__` takes one `rollout_fn`, one `reward_fn`, one `harness` | per-source maps | Token env, World task and judge-graded source in one `Runtime`; every record tagged with its source | P0 | M |
| A2-54 | Per-source staleness limit | Table MiMo | **missing**: global `max_staleness` | per-source τ | Source at τ = 1 never trains staleness > 1 next to a τ = 4 source | P2 | S |
| A2-55 | Per-source length and active-time monitoring | §2.2 MiMo (90× / 66× spread) | **missing**: `StepMonitor.record` is not split by source | — | `reward/mean/<src>`, `len/p99/<src>`, `active_s/<src>` per step | P1 | S |

## 10. Everything else

| ID | Requirement | Art. | Status / code | Missing | Acceptance | P | E |
|---|---|---|---|---|---|---|---|
| A2-56 | Streamed metrics (jsonl / W&B) and persisted sampled rollouts for offline audit | §1 "observation"; §3.7.3.6 MiMo audits | **missing**: `RunReport` in memory; `m12/gate.py` writes one JSON at the end | sinks | Crash at step k leaves k metric lines and the audit records on disk | P0 | S |
| A2-57 | Engine heartbeat and hang detection; remove or re-add engines; re-issue in-flight calls | §3.1.4 GLM slime; Qwen fault recovery | **partial**: `weightsync` subgroups and elastic join; breaker; `EnginePool.generate` has no timeout or health check | health; rerouting | Kill one engine: its calls are re-issued within T; `engine/failures` logged | P1 | M |
| A2-58 | A verifier patch invalidates buffered rollouts on resume | §3.7.3.3 LongCat (fixed at step 80, resumed) | **partial**: content-hash task versions; no link to training state | verifier version on tags | Old-version rollouts re-graded or evicted on resume | P1 | M |
| A2-59 | REMAX and REINFORCE++ estimators | §3.5.3.7 Hunyuan | **partial**: `objectives.group_advantages` (mean, rloo), GAE | two estimators | Golden values | P2 | S |
| A2-60 | Reasoning-effort tag, per-effort length penalty, effort-homogeneous groups | Table DeepSeek; Kimi | **missing** | — | Groups never mix efforts | P2 | S |
| A2-61 | Repetition penalties: zlib ratio (Ling); repeated tool calls (MiMo ToolRep) | Table; §6 | **missing** | — | Golden zlib value; k identical calls penalised | P2 | S |

---

## 11. Prioritized build list

**P0, in dependency order.**
1. **A2-56 sinks:** everything after must be observable. S.
2. **A2-15 group grading stage:** the structural fix that GAR, tournaments and hack zeroing need. M.
3. **A2-16 GAR.** S.
4. **Online hack defence:**
   - A2-28 interception at `World._call_tool` (S);
   - A2-31 audit sample wired into the runtime (S);
   - A2-29 two-level detector, which finally produces `hack/confirmed` (M).
5. **A2-53 per-source routing in one Runtime:** a prerequisite for A2-11, A2-14, A2-54 and A2-55. M.
6. **A2-1 StateStore:** finish and commit the `train/state.py` draft. It also closes checklist #1; A2-2 and A2-58 build on it. L.
7. **A2-9 + A2-10 teacher pool and OPD loss:** route ④ is a whole paradigm with no support today. L + M.

**P1, in order.**
- **Cheap wins:** A2-24, A2-25, A2-26, A2-55 (each S).
- **Context:** A2-46, A2-47, A2-48.
- **Multi-agent:** A2-50, A2-52.
- **Noise:** A2-38, A2-40.
- **Sandbox state:** A2-5, A2-6, A2-8.
- **Hack defence:** A2-30, A2-34, A2-35.
- **Rewards and distillation:** A2-14, A2-17, A2-18, A2-21, A2-11, A2-12.
- **Scale:** A2-42, A2-57.
- **Last, the large items:** A2-4, A2-41, A2-44, A2-32 (each L).

**P2.** A2-3, 7, 13, 19, 20, 22, 23, 27, 33, 39, 43, 45, 49, 51, 54, 59, 60, 61.

**Totals.** 61 requirements:
- **Done:** 2.
- **Partial:** 29.
- **Missing:** 30.
- **P0:** 12, of which 2 are done.

The open P0s fall in four places:
- the state plane (A2-1);
- route ④ (A2-9, A2-10);
- the group-level reward stage (A2-15, A2-16);
- online hack defence (A2-28, A2-29, A2-31).
A2-53 and A2-56 are cross-cutting P0s that also enable later items.
