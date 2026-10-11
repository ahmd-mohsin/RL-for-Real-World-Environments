# M13 bench, M14 packs, M15 studio, RLM: improvements beyond study 01 (2026-10-11)

Licences:
- MIT: verifiers, karotte.
- Apache-2.0: research-environments, SkyRL, Harbor, repo2rlenv, NeMo-Gym, FineEnvs.
- BSD-3: OpenEnv.
- GitSwarm, autoresearch and the articles: pattern only.

**Main finding.** No checkout has Crucible's step metrics, IRT, sealing or contamination tooling. The new material is:
- **the harness as a variable:** sensitivity, harness mixtures, parity;
- **content-addressed release and publish flows;**
- **graph-structured traces,** where a compaction branch is trainable only if the run resumed from it;
- **RLM safety caps;**
- **measuring whether a model uses the RLM scaffold.**

## M13 bench

| # | Change | Source | I / E |
|---|---|---|---|
| 1 | Seal content, not ids: `splits.seal` hashes (id, content digest) pairs, and `bench.digest.task_digest(dir)` is added. A task edited after sealing is not detected today | Harbor `publisher/packager.py:83-95`, `models/dataset/manifest.py:245-262` | H / S |
| 2 | `metrics.harness_profile`: a policy × harness × task matrix with the harness gap, rank-reversal flags, and a **format-failure rate** separate from degraded score. Split IRT θ into a policy term plus a harness offset. The same model moves 13 points across harnesses | `07-articles/hf-space-fineenvs-multi-harness-rl.md:246-300`; FineEnvs `05-multi-harness-rl/eval/evaluate.py:187-256` | H / M |
| 3 | `crucible bench parity`: run arrays, mean ± SEM, an equivalence test (TOST) for M11 export parity | Harbor `adapters/*/parity_experiment.json`, `ADAPTER_CONTRIBUTING.md:309-345` | H / M |
| 4 | `bench/leaderboard.jsonl` keyed by (seal, eligible subset, policy@harness); refuses to rank rows across seals; optional push with loss accounting | verifiers `v1/utils/platform.py:47-128` | M / S |
| 5 | `crucible bench rescore <run>`: re-grade saved episodes, tag them offline, compare against the original | verifiers `v1/cli/replay.py` | M / M |
| 6 | A misbehavior / hack-rate column | karotte `student_misbehavior.py` | M / S |
| 7 | Weight-0 judge criteria as diagnostic columns | research-environments `judges/swe/rubrics.toml` | L / S |

## M14 packs

| # | Change | Source | I / E |
|---|---|---|---|
| 1 | `pack check --level fixture --update-expected` (the fixture level is **done 2026-10-11**; `--update-expected` is not) | NeMo-Gym `environment/_verifier_runner.py:28-32` | H / S |
| 2 | `crucible pack lock`: per-task and per-pack digests in `pack.lock`; `check` warns on drift; the bench seal records the lock | Harbor `packager.py:83-95` | H / S |
| 3 | `crucible pack release|publish`: an immutable staging dir, `release-files.json`, a receipt state machine, an HF push, a remote file-list check, tags; gated on checks passing and no TODOs | repo2rlenv `campaigns/release.py:95-456`; NeMo-Gym `publication.py:61-90`; Harbor `publisher.py:221-241` | H / M |
| 4 | A generated dataset card: licence, canary, families, reward kinds, IRT b as difficulty, splits, seal, composition | repo2rlenv `hub.py:68-115,182-385` | M / S |
| 5 | A `harnesses:` manifest field, so `bench run` runs the matrix | verifiers `configs/gsm8k_rlm.toml` | M / S |
| 6 | `crucible pack catalog build|discover` over committed metadata, with a snapshot digest | OpenEnv `cli/commands/catalog.py:24-82`, `discovery/producer.py:241` | M / M |
| 7 | `crucible pack import harbor://… / hf://…` | repo2rlenv `hub.py:610-830` | M / M |
| 8 | A long-context RLM pack family | research-environments `environments/long_context/` | M / M |

## M15 studio

| # | Change | Source | I / E |
|---|---|---|---|
| 1 | Tool cards stored by hash in M10 runs (`tools/<sha>.json`, the hash on each step), so `label --run` fills `tools` | verifiers `v1/graph.py:318` | H / S |
| 2 | `label --run` for RLM and compacted episodes: one SFT row per branch (segment or sub-call) from `model_calls`; drop compaction branches the run did not resume from | verifiers `v1/trace.py:516-552` | H / M |
| 3 | A `harness` provenance column; `--stratify harness` / `--max-per-harness` (SFT on purpose mixes harnesses) | `hf-space-fineenvs-multi-harness-rl.md:354-398` | M / S |
| 4 | `label --push hf://repo` with a card and a receipt | Harbor `utils/traces_utils.py:1138` | M / S |
| 5 | An explorer index as an immutable SQLite snapshot with a shrink guard | FineEnvs `tooling/hf-rl-explorer/app/indexer.py` | L / M |

## RLM agent and harness

| # | Change | Source | I / E |
|---|---|---|---|
| 1 | **Caps in `RLMConfig` (in the digest):** `max_sub_calls`, `max_batch`, `max_concurrent_sub` (a semaphore around `llm.batch`, which today is a bare `asyncio.gather`), `max_tree_tokens`. When exceeded, the host returns an error string and the run gets a typed stop `sub_budget`. **Without these, RL cost is unbounded** | verifiers `v1/harnesses/rlm/harness.py:76-113`; OpenEnv `envs/repl_env/recursive_backends.py:46-62` | H / S |
| 2 | Sub-model identity plus `sub_trainable` in `RLMConfig` and the harness id. A frozen or external sub-model's calls are non-trainable environment calls, not `capture_incomplete` | SkyRL `examples/train/rlm/rlm_config.py:20-23` | H / S |
| 3 | A compaction branch is trainable only if the next segment actually starts from it; `max_compaction_attempts`, then a typed `compaction_failed` stop | verifiers `trace.py:516-552`, `harness.py:48-67` | H / M |
| 4 | **Scaffold-use metrics** from the kernel journal: World calls per cell, loop cells, truncation hits, re-fetch rate, delegation rate, finish rate. In M13, never in the reward. Optional weight-0 rubric criteria | research-environments `judges/rlm/rubrics.toml`; SkyRL `evidence_rewards.py:103-130` | H / M |
| 5 | A coding-World tool bridge in the kernel (`exec/read/write/search/apply_patch` proxies over `ExecLauncher`) | Harbor `agents/dspy_rlm.py:62-178` | H / M |
| 6 | An RLM long-context bench family (HMAC-generated aggregation, needle and verbatim tasks on an 8k → 1M ladder; data in a kernel variable); RLM vs native, compaction on vs off | research-environments `long_context/*`, `longbenchpro/taskset.py:1-14` | M / M |
| 7 | A harness mixture in M12 (one harness per GRPO group; bench #2 measures transfer) | FineEnvs `train/multi_harness.py:73` | M / M |
| 8 | Depth-2 `rlm.spawn` in the policy with child kernels; persistent child handles in `RLMAgent` | SkyRL `skyrl_gym/envs/rlm/env.py:323`, `rlm_generator.py:185-222`; `pi-on-the-nature-of-the-swarm.md:51-63` | M / L |
| 9 | System-prompt appends per root, sub-agent and leaf | verifiers `harness.py:95-123` | L / S |
| 10 | GEPA-proposed harness edits behind the existing same-seed A/B; a keep/discard/crash ledger; a cost for growth in the digest's size | verifiers `gepa/*`; autoresearch `program.md:37,66-87` | L / M |
| 11 | A content-addressed node DAG for `rlm.<actor>.json` | verifiers `graph.py:83-330` | L / M |

**Live evidence, 2026-10-11:** Claude Sonnet 5.5 under the RLM harness solved the 12-ticket backlog (reward 1.0, $0.35),
where the native harness scored 0.67 (n = 1 each).

**Do not adopt:**
- OpenEnv `REPLRubric`'s per-step process reward and −0.1 for no answer (it conflicts with P16);
- FineEnvs' tool-call-count bonus.
