# M1 taxonomy 07: capability matrix (what Crucible can and cannot express)

Merged from the six read-only studies of the local corpus (`01`–`06` in this folder) and the code inventory (`00`).
Corpus: 30 codebases in `RL-Environments/06-frameworks-and-env-code`, the `07-articles`, the `08-hf-rl-env-datasets`
cards, and the MiMo v2.6 release. Status words: **yes** (Crucible expresses it today, with code named in `00`),
**gap** (expressible after the named primitive), **no** (no route today).

## 1. Environment families

| # | Family (examples; size in the corpus) | Crucible | Missing primitive | Owner |
|---|---|---|---|---|
| 1 | Single-turn verifiable answers: math, MCQ, science, instruction following, long context, safety (Prime 60+ envs, NeMo-Gym ~80 servers, Harbor QA adapters, every trainer's math env) | **no** | single-turn task type (prompt, completion, checker; no tools or sandbox), output parsers, weighted rubrics, pinned judges | M1, M7 |
| 2 | Code generation with hidden unit tests (Harbor HA-5, NeMo F3, Prime F6) | gap | single-turn-to-execution fast path; a code-execution service App | M1, M7 |
| 3 | Repository-level SWE: repair, features, test writing (Harbor HA-1/2, SWE-smith 52k, R2E, NeMo F4, Prime F9) | yes | separate verifier environment with an artifact handoff; F2P/P2P report parsers as a shared grader; bug-injection operators | M1, M7, M8 |
| 4 | Terminal tasks (terminal-bench 2.x, Terminal-Lego, Long-Horizon-Terminal-Bench) | gap | per-family verification of the in-place Harbor runner; continue-until-timeout | M1, M11 |
| 5 | Performance and optimization with continuous rewards (algotune, gso, kernelbench, frontiersmith) | yes | unbounded rewards are allowed; rlimit-sandboxed scorers | M7 |
| 6 | Enterprise record worlds: ITSM, CRM, HR, finance, workplace (EnterpriseOps-Gym 1,150 tasks, MiMo general 266, workplace_assistant, AWM-1K, tau) | gap | seeded database per run, SQL post-state assertions with "other rows unchanged", per-actor identity headers and RBAC, oracle and +N distractor tool modes, an EnterpriseOps-Gym adapter | M1, M7, M11 |
| 7 | Conversations with simulated users (tau2/3, indian_banking, toolsandbox, calendar, Harbor persona tasks) | yes | stop and transfer tokens; LLM-simulated tool results as an actor role; user-side tools | M2 |
| 8 | Search, RAG, deep research (NeMo F8, Prime F11, Harbor judge-graded) | gap | search-index Apps, allowlisted external APIs, pinned judges | M1, M7 |
| 9 | Browser and web (browsergym, WebArena, interactive_browser, web_voyager) | gap | accessibility-tree and DOM observations; multi-site service worlds | M1 |
| 10 | Computer use: desktop, VM, Android (OSWorld 369, computer-1, FineEnvs desktop, androidworld) | **no** | VM or desktop App, keyboard and mouse actions in pixel coordinates, screenshot observations per turn, VM pools with health checks | M1 |
| 11 | Discrete games: Atari, OpenSpiel, chess, TextArena, sokoban, wordle, kuhn, hanabi | gap | native turn order (seats, legal-action masks, simultaneous moves, chance nodes, N players); today only imported OpenEnv two-player games | M1 |
| 12 | Continuous control and simulators: dm_control, CARLA, Unity, SUMO, FinRL, wildfire | **no** | continuous action vectors, mixed continuous and tool actions, raw pixel arrays, simulator ticks advanced by each tool call | M1 |
| 13 | Multimodal input and output: vision QA, image tools, audio (Prime F5, NeMo F14, Harbor describe-image, mmau) | gap | images and audio in messages and instructions (not only tool results); VLM training codec | M0, M1, M12 |
| 14 | Long-horizon research on GPUs: mlgym, rexbench, autoresearch, kernrl (4–72 h) | gap | GPU sandboxes and long horizons exist; continue-until-timeout and per-run metric streams | M1, M10 |
| 15 | Formal proving in Lean (Prime F7 13 envs, NeMo math_formal_lean) | gap | a Lean sandbox App and proof checker | M1, M7 |
| 16 | Multi-agent cooperation and competition (CooperBench, solver-judge, self-play) | gap | per-actor rewards outside imported games, a message bus between policies, multi-agent training records | M1, M12 |
| 17 | Security exploitation (CyberGym 1,507, MiMo cyber) | gap | token-gated result sidecars and per-uid egress rules | M1 |
| 18 | Knowledge-work deliverables: office files, reports (gdpval, MiMo general, job_bench) | gap | tiered, weighted rubric items with gates; evidence extraction with explicit truncation; answer materialization | M7 |
| 19 | Data engineering: SQL, dbt, spreadsheets (spider2-dbt, bird, spreadsheetbench) | gap | DuckDB state adapter | M1 |
| 20 | Live shared services: CRM orgs, GitLab and chat stacks (crmarena, TheAgentCompany) | gap | service reset hooks and a run-serially flag | M1, M10 |
| 21 | Black-box agent harness as the driver (Harbor agents, opencode, claude-code, codex) | yes | capture proxy and harnesses exist; per-rollout harness rotation (harness profiles) | M10, M13 |
| 22 | Next-action verification against demonstrations (NeMo F18) | gap | a step-verifier task type | M7 |

**Count:** 4 families yes, 15 gap, 3 no (single-turn answers, computer use, continuous control). The two largest by
number of environments, single-turn answers and enterprise record worlds, are both gaps.

## 2. Primitives, deduplicated

Each primitive appears in at least two of the six studies unless marked. Severity: **coverage** (blocks a family),
**correctness** (rewards wrong or hackable without it).

| ID | Primitive | Severity | Needed by (families) | Owner |
|---|---|---|---|---|
| P1 | Single-turn task type: dataset row, one completion, checker; no World tools | coverage | 1, 2, 13, 22 | M1 |
| P2 | Separate verifier environment: a fresh box that never ran agent code, fed only declared artifacts, with submission custody (no symlinks, FIFOs or special files; size limits; hash checks of protected assets) | correctness | 3, 4, 17 | M1, M7 |
| P3 | Multi-step tasks: per-step instruction, setup, health check, tests and `min_reward` gating | coverage | 3, 4, 14 | M1 |
| P4 | Seeded state per run from a snapshot, and SQL post-state assertions including "other rows unchanged" | coverage | 6, 19 | M1, M7 |
| P5 | Per-actor identity and RBAC context on every tool call and verifier query | coverage | 6 | M1 |
| P6 | Tool catalog modes: oracle, +N distractors, allow and restrict lists, per-session allow-lists | coverage | 6, 8 | M1 |
| P7 | Native turn order and multi-agent semantics: seats, legal-action masks, simultaneous moves, chance, per-actor rewards | coverage | 11, 16 | M1 |
| P8 | Simulator-owned time advanced per tool call, with idle ticks | coverage | 12 | M1 |
| P9 | Continuous and mixed action spaces with spec export | coverage | 12 | M1 |
| P10 | Multimodal content parts in messages and observations; raw arrays and encoded images; large-message transport | coverage | 10, 12, 13 | M0, M1 |
| P11 | Desktop or VM App with keyboard, mouse and screenshots; VM pools | coverage | 10 | M1 |
| P12 | Remote environment client over a persistent WebSocket session | coverage | 11, 12 | M1, M11 |
| P13 | Per-uid egress allowlist enforced inside the sandbox (today the provider's policy plus an egress self-test) | correctness | 3, 8, 17 | M1 |
| P14 | Health checks, setup hooks, service reset endpoints, a run-serially flag | coverage | 6, 20 | M1 |
| P15 | Dense shaping gated on the outcome (a gate failure zeroes progress) | correctness | all with dense reward | M6 |
| P16 | Rubric trees (gate, weighted sum, sequential), tiers and weights, parsers, judge pinning with verdict errors that mask instead of scoring 0 | correctness | 1, 3, 8, 15, 18 | M7 |
| P17 | Infrastructure-failure conventions mapped to one mask: NeMo-Gym `mask_sample`, MiMo absent reward key, Karotte error events | correctness | all importers | M11 |
| P18 | Bug-injection and environment-breaking operators with "breaks at least one test" validation | coverage | 3 | M8 |
| P19 | Trainer interfaces: text step adapter, per-turn and per-token rewards, named multi-rewards, GiGPO state hash, OpenAI-compatible session endpoint with token capture, termination taxonomy | coverage | training on any family | M10, M12 |
| P20 | Importers: verifiers v0 (109 envs), EnterpriseOps-Gym, Harbor separate-verifier mode | coverage | 1, 3, 6 | M11 |

## 3. Correctness requirements from the articles, against Crucible

From `06` Part 3 (MUST items only).

| Requirement | Crucible |
|---|---|
| Gold 1, no-op 0 for every task | yes: A1, A2 |
| Adversarial probe submissions | yes: A13, A16 |
| Flakiness gate | yes: A4 |
| Infrastructure failure is null, never 0 | yes: masking through runtime and trainer |
| Reward output schema validation | partial: `RewardJson` validates; foreign reward files are mapped per importer |
| Every instruction requirement graded | yes: A7 |
| Hidden verifier assets physically absent | yes: verifier view, CI App, Harbor upload at grade time |
| Separate verifier environment and submission custody | **partial: P2** |
| Unprivileged agent, resource limits, robust tools | yes: guard, pids and memory caps, deadlines |
| Offline by default; allowlists, not denylists | partial: P13 |
| History and future-event scrub | yes: git scrub, temporal masking |
| Shaping gated on outcome; milestones probed with degenerate policies | **no: P15** (approve-all earns progress 0.15 on accounts payable) |
| Group-variance monitoring | yes: `zero_variance_groups` |
| Train and eval cap parity; token-faithful capture; resume-safe ordering | yes for capture and resume; parity is configuration (M12 audit) |
| Images pinned by digest | yes: layer digest check |
| Generation and verification by different models; hack-agent loop per family | yes: forge reviewer, A16 hacker |

## 4. Code worth porting, by owner module

| Owner | Component | Source (license) |
|---|---|---|
| M1 | Multi-step task loop with gating and per-step artifacts | Harbor `src/harbor/trial/multi_step.py` (Apache-2.0) |
| M1, M7 | Artifact manifest and regrade from recorded artifacts | Harbor `trial/artifact_handler.py`, `trial/regrade.py` (Apache-2.0) |
| M1, M7 | Hostile-input-safe submission harvesting; misbehavior versus infrastructure error; grader-only secrets | Karotte `save_submission.py`, `untrusted_paths.py`, `student_misbehavior.py`, `protected_store.py` (MIT) |
| M1 | Per-uid firewall and filesystem hardening | Karotte `confinement.py`, `subprocess.py:297-359` (MIT) |
| M1 | Computer-use action schema to xdotool, coordinate scaling | Harbor `agents/computer_1/runtime.py` (Apache-2.0) |
| M1 | Simulated clock with pause-and-tick and notification drain | Harbor GAIA2 adapter `are_mcp_server.py` (upstream ARE license) |
| M1 | Session lifecycle: close registered before seed, LIFO bounded cleanup, admission control | NeMo-Gym `base_environment_server.py` (Apache-2.0) |
| M1, M11 | Tools-as-Python-files MCP sidecar with result caps and hidden URLs | MiMo `sidecar_entrypoint.py`, `mcp_http.py`, `mcp_bridge.py` (Apache-2.0) |
| M1, M11 | Seed and delete database per run; identity headers; SQL and response-check verifier types; multi-gym routing | EnterpriseOps-Gym `benchmark/{mcp_client,executor,verifier}.py` (Apache-2.0) |
| M1, M11 | Persistent WebSocket environment client; rubric trees; manifest validation | OpenEnv `src/openenv` (BSD-3-Clause) |
| M7 | Library-then-judge cascade; swap-check judge; replay-both-and-diff state grader | NeMo-Gym `math_with_judge`, `equivalence_llm_judge`, `workplace_assistant/utils.py` (Apache-2.0) |
| M7 | Masking through an absent reward key; per-item binary judging with evidence-truncation rule; source-protection fingerprints | MiMo `run_verify.py`, `verify.py`, `_helpers.py:494-604` (Apache-2.0) |
| M7 | Multi-key reward, judge configs, overlay isolation | Harbor `packages/rewardkit` (Apache-2.0) |
| M7, M8 | Graded F2P × P2P; protected-file hash contract; rlimit scorer; history scrub and egress denylist | Repo2RLEnv `_pr_runtime_verifier.py`, `cli_gym/grade.py`, `frontiersmith/grade.py`, `_env_guard.py` (Apache-2.0) |
| M8 | 135 procedural bug-injection operators; break-tests validation | SWE-smith `bug_gen/procedural/`, `harness/valid.py` (MIT) |
| M2 | Strict-plus-dense reward with efficiency tie-break; persona user simulator with stop tokens | NeMo-Gym `indian_banking/core/{reward,user_sim}.py` (Apache-2.0) |
| M6, M12 | GiGPO state hashing and step returns | ROLL `step_env_manager.py`, `agentic/utils.py:60-200` (Apache-2.0) |
| M12 | Per-token reward assignment on captured interactions | AReaL `experimental/openai/cache.py` (Apache-2.0) |
| M10 | Sandbox protocol with warm queue and dead-sandbox replacement | rLLM `sandbox/{protocol,warm_queue}.py` (Apache-2.0) |
| M12 | Token-drift classification (clean, realign, fork) | slime `agent/trajectory.py` (Apache-2.0) |
| M11 | verifiers v0 semantics (SingleTurnEnv, MultiTurnEnv, ToolEnv, Rubric, parsers) | verifiers v0 as used by prime-environments (MIT); the local checkout holds v1 only |

## 5. What this means for the M1 audit

M1 owns P1–P14. In order of reach (environments unlocked per unit of work) and correctness:

1. **P2** separate verifier environment with submission custody (correctness; Karotte and Harbor ports).
2. **P1** single-turn task type (the largest family by count).
3. **P4–P6** enterprise record primitives (Crucible's own target domain; unlocks EnterpriseOps-Gym and MiMo general).
4. **P3** multi-step tasks with gating (Harbor port).
5. **P13** per-uid egress allowlist (correctness; Karotte port).
6. **P14** health checks, reset hooks, run-serially flag.
7. **P7** native turn order and multi-agent semantics.
8. **P10** multimodal messages and observations (with M0).
9. **P12** WebSocket environment client.
10. **P8, P9** simulator time and continuous actions.
11. **P11** desktop and VM App.

Outside M1: P15 (M6, already known: the approve-all finding), P16 (M7), P17 and P20 (M11), P18 (M8), P19 (M10, M12).
