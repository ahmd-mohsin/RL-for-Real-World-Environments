# M13 `bench` and M15 `studio`: what the local frameworks do, and what we adopt

Date: 2026-10-11. Scope: every checkout under `RL-Environments/06-frameworks-and-env-code/` plus `agentenv-framework`,
the articles in `RL-Environments/07-articles/` and the dataset cards in `08-hf-rl-env-datasets/`. Three read-only studies:
metrics and reports; benchmark construction; CLI, datasets and packs.

`B = RL-Environments/06-frameworks-and-env-code`. Licenses checked in the checkouts:
- Apache-2.0: NeMo-Gym, Harbor, prime-rl, verl, rLLM, agentenv, Repo2RLEnv, FineEnvs.
- MIT: verifiers, mini-swe-agent, SWE-smith, karotte.
- BSD-3: OpenEnv.
- ToolSandbox, vendored in NeMo-Gym, has Apple's custom license: **pattern only**.

## 0. Bottom line

**No checkout implements these:**
- progress AUC over Φ;
- first irrecoverable step;
- recovery rate on fault-injected starts;
- in-trajectory regression counts;
- paired significance tests (McNemar, paired bootstrap);
- per-step heatmaps;
- 2PL IRT;
- MinHash/n-gram decontamination;
- a per-model temporal-cutoff filter;
- DPO pairs built from agent traces.

M13 and M15 build these themselves. Everything else has a strong pattern to adopt.

**Several frameworks score infra errors as 0** [ERR=0]:
- Harbor: `utils/pass_at_k.py:41`, `metrics/base.py:25,32`, viewer `server.py:1681,1825`;
- verl: `metric_utils.py:969`;
- rLLM: `eval/results.py:74`;
- SWE-smith: `harness/eval.py:74`;
- EnterpriseOps-Gym: `executor.py:514`;
- NeMo-RL: fractional `num_correct`.

Crucible never does this (DESIGN P16). The one framework that gets it right is NeMo-Gym.

## 1. Metrics, aggregation, comparison, reports (M13)

| # | Adopt | Where | How |
|---|---|---|---|
| 1 | **Mask semantics and a coverage block**: masked rows leave quality metrics but stay in coverage (`measured`, `masked`, `measured_tasks`, `fully_masked_tasks`); an all-masked group reports coverage only and never a 0 | NeMo-Gym `reward_profile.py:842-950,1132-1149`; `base_resources_server.py:110-131` | PATTERN (our `RunRecord.scored_reward` already masks) |
| 2 | **pass@k and pass^k** (pass^k: all k succeed, tau-bench reliability), unbiased via `math.comb`, measured samples only; non-binary rewards via a threshold, or the unbiased **expected max of k** from order statistics (never NeMo-Gym's order-biased `max(vals[:k])`, `:592`) | prime-rl `orchestrator/utils.py:95-115`, `orchestrator.py:916`; rLLM `results.py:27-45` | PATTERN, own code |
| 3 | **Uncertainty**: within-task spread vs across-run spread, with t-CIs and a warning on unequal coverage; bootstrap CIs for non-normal metrics (AUC, cost per success) | NeMo-Gym `reward_profile.py:226-353,661-689`; verl `bootstrap_metric` `metric_utils.py:805` | PATTERN; seeded bootstrap |
| 4 | **Flip analysis and a compare report** (binary vs continuous, pass→fail / fail→pass, regressions first, md + json). **Upgraded**: join on (task, sample index), not row index; **paired bootstrap over tasks** and **exact McNemar** on flips instead of NeMo-Gym's unpaired Welch | NeMo-Gym `comparison/diff.py:205-359`, `report.py:249-345` | PATTERN plus new statistics |
| 5 | **Milestone DAG mapped to the step reached**: stage reach, the step at which each stage was first reached, and a minefield that nullifies the run | ToolSandbox `evaluation.py:984-1237` | PATTERN only (license) |
| 6 | **Gate profile**: per-verifier passed/total/pass_rate, micro `verifier_level_pass_rate`, macro average | EnterpriseOps-Gym `executor.py:580-606`; OpenEnv `rubrics/base.py:164-185`; Harbor per-key aggregation `metrics/base.py:30` (minus the zero-fill) | PATTERN |
| 7 | **Probe floor**: score the untouched task; a no-op that passes is invalid | verifiers `cli/validate.py:242-265` | PATTERN (our A2 and M9 do this; the bench reports it) |
| 8 | **Dual headline**: the measured mean, with the raw mean (errors as 0) in parentheses, so masking is visible; only trainable actors counted | verifiers `utils/format.py:57-72`, `dashboard/eval.py:335` | PATTERN |
| 9 | **Cost and perf summary**: field-presence-gated stats plus `observability_coverage`; cost per success = all spend (errored included) / measured successes | NeMo-Gym `compute_perf_summary` `:985-1063`; Harbor `trial/result.py:100-136` | PATTERN |
| 10 | **Per-step tally** for the run × step matrix | agentenv `explorer/routers/runs.py:299-326` | PATTERN (its CANCELED→failed fold, `:356`, is the bug we avoid) |
| 11 | **A single self-contained HTML report** (inline CSS, no build step, archivable); reward–cost Pareto and per-trial waffle views | rLLM `eval/visualizer.py:183-912`; Harbor `job-evals-pareto.tsx`, `job-evals-waffle.tsx` | PATTERN |
| 12 | **Result layout**: rollouts + aggregate + failures sidecar + per-trial dirs | NeMo-Gym `<stem>.jsonl`, `_aggregate_metrics.json`, `_failures.jsonl`; Harbor `jobs/<job>/<trial>/` | PATTERN: `bench/<name>/{summary.json, episodes.json, report.html, compare/}` next to the M10 run journal |

## 2. Benchmark construction (M13)

**Splits:**
- **Never split by row.** NeMo-Gym `xlam_fc/generate_dataset.py:151` and FineEnvs `train_sft.py:73` use `train_test_split(seed=42)`, which leaks siblings across splits. Split by a group key: the lineage root plus the axis group.
- **Assignment** is a salted HMAC over the key: stable as items are added, and builders cannot predict it. This is PATTERN from FineEnvs (`build.py:78-87`, sha1 ordering; `build.py:93-97`, a sealed manifest) and prime-agent (`ci_test_shard.py:97`).
- **Time axis:** hold out contiguous blocks with a guard gap (FineEnvs `dock.py:31-62`).
- **Seal:** `sha256(sorted bench ids ‖ salt commitment)`, written before builders see test traces (DESIGN §11.2).
- Showcase tasks (01–04) never enter a published split (ARCHITECTURE §6).

**IRT (2PL):**
- No checkout has it. The closest is NeMo-Gym gdpval's **anchored Bradley–Terry**: the reference ratings are fixed and the one free rating is found by bisection on a monotone gradient (`resources_servers/gdpval/comparison.py:2046-2119`). Coverage floors keep anchors in partial fits (`multistage_elo.py:62-70`).
- **Model:** P = σ(a(θ − b)), fitted by MAP with priors θ ~ N(0,1), b ~ N(0, 2²), log a ~ N(0, 0.5²). The data are binomial counts (respondent = model × harness).
- **Linking** is **fixed-anchor calibration**: freeze the anchors' (a, b) and estimate fresh items and new θ against them. This is the gdpval idea, and better than mean–sigma at our 10–30 respondents.
- **Rasch** (a = 1) below about 15 respondents.
- **Drift:** refit the anchors freely and quarantine any with |Δb| > 2·SE. Drift usually means contamination.
- θ by bisection, with SE = 1/√Σ n a² p(1−p).
- **Next item** by maximum Fisher information.
- **Item QA:** reject a < 0.2.

**Contamination:**
- **Canaries:** a per-bench-version GUID, Harbor style (`cli/init.py:168-200`, stripped before the agent sees it at `task.py:9-28`), on top of our per-task canary (`accept/checks/core.py:197`).
- **Exact overlap:** 13-grams on normalized words (GPT-3), and a 50-character normalized substring for code (StarCoder).
- **MinHash LSH** for corpus scale: 5-word shingles, 128 permutations, 16 bands × 8 rows (threshold about 0.71). Candidates are verified by exact Jaccard ≥ 0.8.
- **Structural fingerprints:** a file tree of path + git blob sha1 (FineEnvs `general_fetch.py:18`).
- **Leak regex** for issue and PR references (SWE-rebench-V2 card `README.md:186-188`).
- Repo2RLEnv's phrase list (`_oss_instruct.py:196-229`) is too weak to adopt.

**Temporal decontamination:**
- An item is eligible for a model iff `source_time ≥ cutoff + margin`, half-open (NeMo-Gym `benchmarks/livecodebench/prepare_utils.py:127-131`).
- Undated items are excluded (research-environments LiveCodeBench `taskset.py:178`).
- Report the eligible subset per model, and compare models by IRT θ, because raw pass@1 over different subsets is not comparable.
- **Schema gap:** Crucible has no `source_time` yet.

**Dynamic items:**
- **Seeds:** `HMAC(K_version, family/seed_ref/idx)` drives a string-keyed `random.Random` (FineEnvs `dock.py:92`). **Never** seed + i, which collides (prime-environments `enigmata.py:362`, research-environments `uuid_ctf/taskset.py:281`). Never a global `random.seed` (SWE-smith).
- **Commit–reveal:** publish sha256(K) when a version ships, and reveal K when it retires.
- **Validation chain:**
  1. gold passes twice (flakiness; SWE-rebench-V2 card `:195`);
  2. no-op below solved (verifiers `validate.py:52`; OpenEnv RFC 008 `:182-185` measures the floor);
  3. naive and greedy baselines do not solve it (FineEnvs `dock.py:287`, `build.py:43`);
  4. contamination scan;
  5. the ladder band;
  6. IRT placement against the frozen anchors.
- **Composition:** about 30% anchors and 70% fresh, stratified by family, horizon and b quantiles. Top up with FineEnvs' seed loop (`build.py:64-77`).

## 3. CLI, datasets and packs (M15)

**CLI.** Stay on click: Crucible and agentenv already use it.
- **Lazy subcommand table:** rLLM `_LazyGroup` (`rllm/cli/main.py:19-113`) plus NeMo-Gym's declarative `COMMANDS` (`cli/main.py:746`), so `crucible --help` stays fast over 13+ heavy verbs.
- **User errors print one line**, and `--verbose` shows the traceback: agentenv `_UserErrorsAreNotCrashes` (`cli/__init__.py:28-43`, PATTERN).
- **Run directory:**
  - `configs/resolved/<cmd>.json` + `results.jsonl` + `summary.json` (verifiers `cli/output.py:14-122`);
  - `--dry-run` on mutating verbs (Harbor `jobs.py:1275`);
  - `--json` on reporting verbs, with plugin import noise kept on stderr (agentenv `_cli.py:225-228`);
  - skip-existing by key (mini-swe-agent `swebench.py:229-232`).

**`crucible label`.** The formats are TRL-canonical JSONL (`infra-huggingface-trl/docs/source/dataset_formats.md`):
- **SFT** `{messages, tools}`, with tool calls preserved and tools as JSON (`:82-215`);
- **DPO** `{prompt, chosen, rejected, tools}` with an explicit prompt (`:334-352`);
- **KTO** `{prompt, completion, label}` (`:362-372`).

Dense credit rides along:
- as prime-rl's per-message **`message_loss_mask`** (`trainer/sft/data.py:289,367`), which works in TRL, prime-rl, NeMo-RL (`messages`) and verl (`messages_key`);
- plus a per-message `weight` column.

Curation:
- rLLM `dataset from-eval --filter "0 < avg < 1" --select correct|best|best-n|all --max-per-task N` uses a safe AST filter DSL (`rllm/cli/dataset.py:304-415`, `eval/filter_dsl.py:46-66`). We use the same semantics with a closed expression grammar.
- Provenance columns follow Harbor's trace rows (`utils/traces_utils.py:18-45`) and SWE-smith `collect_trajs.py:67-74`: task, run, episode, model, reward, resolved, a content-hash trajectory id.
- Multimodal input raises rather than being dropped (Harbor `:48`).

**No framework builds DPO pairs from agent traces offline.** Ours are Crucible-native:
- **best vs worst of k** on the same task and seed, with a reward margin, and the chosen one must pass (the online logic of TRL online DPO, `online_dpo_trainer.py:1253-1288`);
- **correction pivots**: a step graded wrong and then corrected (`signals/pivots.py`), with the prompt the shared prefix, rejected the wrong action, chosen the fix.

**`crucible pack new / check`:**
- **Scaffold:**
  - NeMo-Gym's conflict-safe writer (`environment/scaffold.py:909-950`): plan every file, identical = no-op, conflict = error before any write, no symlinks, `open("x")`;
  - OpenEnv's `__NAME__` token substitution in contents and filenames (`cli/commands/init.py:213-299`);
  - `TODO` placeholders that a check rejects (NeMo-Gym `publication.py:38-58`);
  - the template version recorded for a later 3-way upgrade (karotte `create_env.py:18-35`).
- **Manifest:** a pydantic model with a cross-field contract and an exported JSON Schema (NeMo-Gym `manifest.py:215-314,386`; OpenEnv `validation/manifest.py:39-242`).
- **Check levels:**
  - static: schema, layout, placeholders, entry point;
  - fixture;
  - semantic: model-free **gold passes, no-op fails** (verifiers `validate.py:237-370`; OpenEnv `validate --level`).
- **Reporting:** a versioned severity policy (OpenEnv `validation/policies/severity-v1.json`); exit codes 0 pass/warn, 1 fail, 2 unsupported, 3 internal (OpenEnv `validate.py:150-160`).
- **CI:** `pack check --all --json` fails when an installed pack fails to load (agentenv `plugin check`, `cli/plugin.py:352-390`).

**README.** A zero-dependency first command (agentenv `README.md:17-25`, `agent-env run hello`), then one linear path (karotte `README.md:24-40`). Every command prints the next one (vf-init `init.py:235`).

**UI, later.** An agentenv `ExplorerPlugin` (`explorer/plugin.py:23-37`) under `/api/v1/crucible`, with routes shaped like Harbor's viewer API.
