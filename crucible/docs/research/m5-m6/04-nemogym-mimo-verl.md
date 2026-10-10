# M5/M6 research 04: NeMo Gym, MiMo v2.6 RL data, XiaomiMiMo verl fork

Scope: reward components and multi-key rewards, `swe_pivot`, verifier fixtures, rollout collection and masking, judges and rubrics, SWE-bench grading and anti-cheat, token capture, re-verification (NeMo Gym); `verifier_meta` rubric items, rule checks and reward composition (MiMo data); the advantage pipeline, turn-span operators, infra masking, rubric judge, anti-hack and multi-harness training (verl fork plus `mimoagent-osr` and `uni_agent`).

Path prefixes used below (all under `RL-Environments/`):
- `NG/` = `06-frameworks-and-env-code/nvidia-nemo-gym/`
- `MD/` = `01-lab-releases/xiaomi-mimo-v2.6-rl-oss/`; `E/` = `MD/general/envs/s3k_0393_consulting_bizops_analytics_en_t2_rl_008/` (a representative env; `verify.py` and `run_verify.py` are byte-identical across all 266 envs, checked by md5)
- `VL/` = `06-frameworks-and-env-code/xiaomimimo-verl/`; `MA/` = `VL/third_party/mimoagent-osr/src/mimoagent/`; `UA/` = `VL/third_party/uni_agent/uni_agent/`

---

## 1. TL;DR: the most useful findings for M5, M6 and harnesses

1. **No dense or per-turn reward reaches a trainer in any of these three sources except token-mark penalties.** NeMo Gym returns one scalar per rollout ("one logical rollout retains one reward and advantage", `NG/responses_api_agents/osworld_agent/README.md:62-64`). Its OpenEnv adapter *sums* per-step rewards into that scalar (`NG/resources_servers/openenv/app.py:227-243`). The verl fork's only per-turn operators mark a span and then mask it, subtract from it, set it, or sign-rebalance it (`VL/verl/trainer/ppo/v1/trainer_base.py:102-163`, `VL/verl/trainer/ppo/signed_rebalance.py`). **Crucible should** build the additive per-turn shaping operator itself. Nothing in these sources implements it.

2. **DESIGN.md §6.4 (line 429) overstates `arvo_penalties.py`.** That file does not add turn-level shaping. It (i) applies a scalar, group-relative length penalty to the final reward (`VL/verl/trainer/ppo/arvo_penalties.py:100-144`) and (ii) marks the `llm_turn_spans` of turns with tool-format errors with κ, then runs sign-aware mass-conserving rebalancing (`:146-190`). It also has a latent bug. `apply_tool_penalty` passes 4 positional arguments (`:183-190`) to `rebalance_dense(advantages, hit, response_mask, *, ...)`, which accepts only 3 (`signed_rebalance.py:175-183`). The `arvo_penalties.enable` path therefore raises `TypeError` as written. **Crucible should** fix the DESIGN wording, and port only the span-validation and mass-conservation patterns from this file.

3. **The verl hook point for A_t is exact.** It goes in `PPOTrainer._compute_advantage`, after `compute_advantage_for_multi_trajectories` (`trainer_base.py:2053-2063`). That function computes GRPO on each session's *final* output and broadcasts one scalar to every row of the session (`VL/verl/trainer/ppo/v1/utils.py:204-277`). It belongs beside the existing post-GRPO operators (`:2102-2150`), and must update `returns` too (`:2135-2136`). Data plumbing already exists: agent loops ship `llm_turn_spans` in `extra_fields` (`VL/recipes/arvo/agent_loop.py:245-247`, `trajectory_metadata.py:22-40`). uni_agent projects `generation_spans` into a `turn_index` tensor and fails closed (`UA/framework/framework.py:479-503`). **Crucible should** add a `turn_shaping` list (one float per span) next to `llm_turn_spans`, and apply it with the validation from `arvo_penalties.py:163-171`. Section 5a gives the algorithm.

4. **Injecting net negative advantage mass is a documented failure.** "A whole-turn `adv_set -1` instead injects net negative mass into the batch, which with no KL / entropy bonus flattens the policy and lengthens outputs — the documented reason the reference RL framework replaced it with this" (`signed_rebalance.py:27-29`). **Crucible should** center the shaping term with the *prompt-mean token weights* (`core_algos.compute_prompt_loss_weights`, used at `trainer_base.py:2121-2125`), so that it adds zero net mass per group. It should also log `adv_pos_sum/adv_neg_sum` pre and post, as `signed_rebalance.py:158-171` does.

5. **Dynamic sampling will silently remove all-fail groups that dense credit is supposed to rescue.** In the MiMo profile `filter_groups.metric=reward` (`VL/recipes/code/config/train.yaml:160-163`), groups are evicted when the non-infra *scalar* reward has zero std (`VL/verl/trainer/ppo/v1/replay_buffer.py:257-269`). **Crucible should** put the gated scalar R (including κ(1−O)Φ) into `rm_scores` so that partial-credit variance survives the filter. Per-turn shaping alone never does.

6. **The MiMo recipes use Dr.GRPO, so A^GRPO is not std-normalized** (`norm_adv_by_std_in_grpo: false`, `train.yaml:154`; required by `arvo_penalties.py:79-90`). **Crucible should** scale the shaping term by the group outcome std (or tune η in reward units), not add a unit-variance z-score to an unnormalized outcome advantage.

7. **`swe_pivot` is an action-equivalence verifier, not a decision-point detector.** Pivot detection happens offline. An LLM judge (`judge_pivots.py`, "N runs majority vote, 6 failure patterns", about 6% of turns) is cited in `NG/resources_servers/swe_pivot/README.md:13-43` but not present in the repo. Selection is then by on-policy reward variance: at least 8 local rollouts per candidate; keep mixed groups; drop all-0 and all-1 groups; drop the easiest first (`NG/.agents/skills/nemo-gym-pivot-datasets/references/config-training-and-agent-ref.md:136-146`). **Crucible should** define a decision point as a step whose K-continuation outcomes are mixed. That is exactly C3, and it gives M5 a stage-boundary candidate and M6 the states where credit matters.

8. **`swe_pivot`'s layered comparator is a ready-made "does this action match the expert's action" check** for aligning a policy run to a recorded trace, and for imitation-at-pivot rewards. The layers are: tool category (63 names, 7 categories); target match (path suffix, editor sub-command, view-range overlap ≥0.8, bash verb + paths + similarity floor 0.3); then `min(sim(old_str), sim(new_str))` with SequenceMatcher capped at 2,000 characters (`NG/resources_servers/swe_pivot/app.py:54-465`). **Crucible should** PORT it for M5 trace alignment and tracker "deviation" detection.

9. **ToolSandbox's milestone/minefield DAG matcher is the closest existing implementation of M5 milestone mapping.** Each milestone is a set of snapshot constraints scored by geometric mean. Milestones are mapped to snapshot indices monotonically, consistent with a topological order of the milestone DAG (DFS with heap and pruning). The final score is `similarity = 1[minefield==0] · mean(milestone similarity)` (`NG/resources_servers/toolsandbox/tool_sandbox/common/evaluation.py:726-1302`). **Crucible should** port the data model and the matcher for offline milestone-to-step mapping, and derive an incremental greedy version for the online tracker.

10. **The MiMo `verifier_meta.json` item schema is the C4 rubric-item schema to adopt.** Fields: `{id, tier: critical|important|sanity, method: llm|rule, question, pass_anchor, files, fn, ground_truth, weight, dim, tolerance}`. Across 266 envs there are 1,427 items: tiers 847/491/89, methods 1,212 llm / 215 rule, 2-10 items per env, mean 5.4. Gates (`gate`) and source protection (`src_protect`) are supported by the grader but appear in none of the shipped envs (computed over `MD/general/envs/*/verifier_meta.json`). **Crucible should** adopt `pass_anchor`, which holds the ground truth plus explicit accept and reject equivalences. It is the main reason these binary LLM items are reproducible.

11. **Adopt both judge prompts.** MiMo's text judge is strict 0/1 with no partial credit and judges evidence only. If the target content falls in a truncated or unextracted region it scores 0, with "evidence truncated" stated in `why` (`E/verify.py:86-99`). It judges each item in a separate call (`:311-340`). mimoagent's *agentic* judge runs inside the doer's final environment, judges "AS DELIVERED", keeps any mutation reversible, and delivers verdicts as a file (never parsed from chat). Missing or invalid verdicts are an infra fault (`MA/environments/rubric_judge.py:92-126, 422-454, 579-611`). **Crucible should** use the text judge for cheap per-checkpoint items and the agentic judge for expensive end-of-episode items.

12. **Use one masking convention throughout: "a failed measurement is not a zero".** It recurs in every source. NeMo Gym uses `mask_sample` plus `failure_kind` plus a failures sidecar (`NG/nemo_gym/base_resources_server.py:107-148`, `judge.py:85-121`). The verl fork uses `is_infra` plus the -999 sentinel, replaced in GRPO by the valid-group mean with advantage forced to 0 (`VL/verl/trainer/ppo/core_algos.py:306-351`). MiMo writes `reward.json` *without* a `reward` key, which the trainer reads as `reward/testbed_corrupted` and masks (`E/run_verify.py:102-151`). One data point: a -999 left in a GRPO group "collapsed the measured pass/fail advantage gap from 2.15 to 0.004" (`VL/tests/recipes/design/test_bridge_contract_on_cpu.py:285-286`). **Crucible should** make C6 a typed `(mask: bool, failure_kind, failure_stage, failure_reason)` record, impute the group mean for masked rows, and *exclude masked rows from the group statistics*, as in `trainer_base.py:2023-2040`.

13. **Judge failure is infra, but the sources differ on unparseable output.** NeMo Gym scores a received-but-unparseable judge reply as a legitimate wrong answer (`NG/nemo_gym/judge.py:25-27`). MiMo retries the parse twice, then masks (`E/verify.py:329-337`). **Crucible should** mask after N parse retries, alarm when the unparseable rate exceeds a threshold, and log the judge input so that reward hacking through the judge stays detectable.

14. **Anti-hack lessons for dense rewards.**
    - **Block, don't terminate.** Return a plausible dummy observation (`"Permission denied. Resource unavailable."`) instead of killing the rollout. Killing it "destabilizes training when whole rollouts are rejected" (`MA/agents/antihack.py:1-40, 198-215`). The block patterns came from 6,026 judged hacking trajectories (`:86-89`).
    - **Penalize supersets.** Score every surplus element 0, so that copying the whole candidate pool scores near 0 (`E/_helpers.py:461-468`). Use F1 for multi-call actions (`config-training-and-agent-ref.md:107-110`).
    - **Missing arguments are not a match.** A vLLM parser bug that dropped the `command` argument used to score full credit in `swe_pivot` (`app.py:351-356`).
    - **Watch threshold ceilings.** The word-overlap similarity has a hidden ceiling of 0.5, so any threshold above it rejects everything (`config-training-and-agent-ref.md:62-71`).

    **Crucible should** turn these into R4/R5-style property tests for every C-operator.

15. **Use state-hash collateral checks as sanity items.** MiMo's `check_no_unrelated_mutation` SHA-256-hashes every non-target row with the mutable columns removed, and compares it to a pinned digest (`MD/general/envs/s3k_0018_accounting_audit_tax_en_t2_rl_004/verifier_meta.json`, `check_code`). `protect_diff` fingerprints files and DB tables outside the legal write footprint (`E/_helpers.py:536-560`). **Crucible should** auto-generate one such "no collateral damage" check per milestone from the recorded write footprint (M4 last-writer attribution). It is a cheap per-step check, and it implements "reward = f(state)" (R6).

16. **Harnesses.** MiMo trains with 4 white-box harnesses mixed deterministically: `mini-mimocode`, `mini-bash`, `mini-claude-code`, `mini-codex` (`VL/config/agent/code/mix-four-whitebox.yaml`). The mixing mode is per prompt, per rollout or per step-hash (`VL/recipes/code/mimoagent_runner.py:109-137`). Advantages can optionally be grouped per harness (`trainer_base.py:2046-2063`). NeMo Gym has adapters for claude_code, codex, opencode, openclaw, pi, hermes, terminus-2, mini-swe-agent, harbor and verifiers (`NG/responses_api_agents/*`), plus a TE-1..TE-9 evidence-conformance checker (`NG/nemo_gym/harness_capabilities/README.md`). **Crucible should** (i) record `agent_type` on every trajectory, (ii) use "paired" mixing (harness varies within a group) with per-harness subgroups for C3 calibration, and (iii) gate harness admission on an evidence-conformance check.

17. **Segment one step as one model call; treat reasoning as transparent.** Treating reasoning items as turn breaks matched the harness's own model-call count on about 76% of trajectories. Treating them as transparent matched 100%. "Hoisted" reasoning leaked future conclusions into 44% of prefixes in one source (`NG/.agents/skills/nemo-gym-pivot-datasets/references/rollout-artifact-pitfalls.md:9-80`). **Crucible should** adopt these rules in M5 segmentation and its replay audit, and cross-check the step count against the producer's own count.

18. **Verifier fixtures map onto R1/R2.** A NeMo Gym verifier must ship `full_reward`, `zero_reward`, `malformed` and (if seeded) `determinism` cases, and the first two must pin the endpoints of the declared range (`NG/nemo_gym/verifier_fixture.py:163-253`). **Crucible should** DEPEND on this pattern: R1 is the oracle at max, R2 is no-op at 0, the malformed case raises, and determinism is a fresh-server repeat.

---

## 2. NVIDIA NeMo Gym

### 2.1 The verify contract and multiple reward keys

- `BaseVerifyResponse` has `reward: float`, `mask_sample: bool = False`, `failure_kind: Optional[str]` and `failure_reason: Optional[str]` (`NG/nemo_gym/base_resources_server.py:107-148`).
  - `mask_sample` means "the reward is not a valid measurement of the evaluated system": a lost session, an unavailable judge, an OOM-killed container, a reset timeout.
  - It is explicitly *orthogonal* to `failure_kind`. A degraded but validly measured sample keeps `mask_sample=False` while naming a kind (`:110-122`).
- `BaseMultiRewardVerifyResponse` adds `reward_components: dict[str, float]`. Keys must be consistent across tasks. `reward` remains the scalar aggregate. Components must also be exposed as top-level numeric fields to reach the metrics, which do not descend into the dict (`:151-164`).
- Worked example: `correctness`, `schema_valid` and `format`, each in {0,1}, with `reward = sum` (`NG/resources_servers/example_tool_call_multireward/app.py:14-37, 154-163`). The stated reason for components is GDPO: "two responses with the same total reward but different composition receive different advantages (whereas GRPO, which normalizes the summed reward, collapses them)".
- ViBench: `reward_components = {test_plan: normalized_score}`, reward = mean over **all** plans. A build that cannot be seeded counts 0 instead of being dropped (`NG/resources_servers/vibench/app.py:641-646`). A grader *configuration* error is raised, never zeroed: "zeroing it would yield a whole dataset of silent zeros" (`:606-609`).

For Crucible: emit O, p, e, Φ_end and each C-operator value as `reward_components`, plus a scalar R. This lets the verl GDPO estimator be tried as an alternative to the multiplicative gate (§4.2).

### 2.2 Failure vocabulary, routing and masking

- **Vocabulary.** `<domain>_<condition>` names, low cardinality, safe as metric labels (`NG/nemo_gym/failure_kinds.py:52-130`):
  - transport: `transport_unreachable`, `transport_peer_drop`, `transport_timeout`, `transport_local_resource`
  - agent: `agent_run_error`, `agent_request_failed`, `agent_timeout`, `agent_command_not_allowed`
  - judge and verifier: `judge_failed`, `judge_unparseable`, `verifier_error`
  - provider: `provider_unavailable`, `provider_quota_exhausted`, `provider_oom_killed`
  - session: `session_lost`, `session_expired`, `session_release_failed`
  - lifecycle: `cancelled`, `kill_shaped`, `shutdown`
  - persistence and cohorts: `persistence_failed`, `cohort_incomplete`

  Environment-specific kinds take the form `<server>:<kind>`. An unregistered kind produces one warning and is not rejected (`:148-165`). Stages are `admission | seed | agent | verification | cleanup` (`:49`). Retry and mask decisions are properties of the *occurrence*, not the name (`:21-27`).
- **Routing** (`NG/nemo_gym/rollout_collection.py:163-192`):
  - Successes go to the main JSONL.
  - Failures go to a `<stem>_failures.jsonl` sidecar, one row per attempt, tagged `_ng_failure_class`.
  - `kill_shaped` failures (SIGTERM, dead Ray actor, OOM) are written **nowhere**. "The absence of a row is the canonical signal", and resume re-dispatches them.
  - Attempts are capped by `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS` (default 3). `_ng_failure_terminal` rows are never retried (`is_terminal_failure`, `:1243-1284`).
- **Judge failsafe.** `judge_failsafe` wraps `verify()`. A `JudgeError` (transport, timeout, auth or HTTP failure) returns `reward 0.0, mask_sample True, failure_kind "judge_failed"` and routes the row to the sidecar (`NG/nemo_gym/judge.py:85-121`). The boundary is explicit: "a *received-but-unparseable* response is a legitimate wrong answer (let the parser score it, don't raise)". Wrap the call, not the scoring, or ordinary bugs get misfiled (`:25-31`).
- **Metrics.**
  - `select_measured` removes masked samples (and their input rows) from quality metrics, and strips the flag so it never becomes `mean/mask_sample` (`NG/nemo_gym/reward_profile.py:856-891`).
  - Coverage reports `measured_rollouts`, `masked_rollouts`, `measured_tasks` and `fully_masked_tasks` (`:928-950`).
  - In-run progress reports `masked_pct` and `reward_unmasked`. The gap between `reward` and `reward_unmasked` is "the score lost to infrastructure" (`rollout_collection.py:131-159`).
  - Opt-in "count failures as zero" exists for metrics only, and never writes a verdict into the artifacts (`:1667-1706`).
- **Mask policies in agents.** These differ in exactly the cases Crucible must decide:
  - `anyswe_agent`: mask if `(resolved and agent_error_kind in {max_iteration, context_window}) or agent_timed_out or error_kind in {eval_timeout, sandbox}` (`NG/responses_api_agents/anyswe_agent/app.py:103-113`). The resolved-but-truncated case counts as an "accidental reward".
  - `swe_agents`: masks *every* `max_iteration` / `context_window` case regardless of `resolved`, plus eval timeout, agent timeout, agent OOM and eval OOM (`NG/responses_api_agents/swe_agents/app.py:3895-3915`).
  - `anyterminal_agent`: mask on container timeout, agent timeout or sandbox failure (`NG/responses_api_agents/anyterminal_agent/app.py:898-903`).
  - OSWorld `classify_rollout_outcome`:
    - An infrastructure reason or an incomplete evaluation is masked.
    - Horizon exhaustion, explicit DONE/FAIL, environment done, and **an unparseable sampled response** are *valid* outcomes and not masked.
    - Missing termination facts produce `rollout_state_incomplete`, which is masked (`NG/responses_api_agents/osworld_agent/rollout_outcome.py:74-125`).
- **Evidence rule.** A masked result must still carry a finite numeric `reward` plus a non-blank `failure_kind` and `failure_reason` (TE-6, `NG/nemo_gym/harness_capabilities/README.md:50-56`).

### 2.3 `swe_pivot`: the exact algorithm

**Pipeline** (`NG/resources_servers/swe_pivot/README.md:13-43`):
1. 74,925 SFT trajectories.
2. "Difficult patches": ≥2 files and >25 changed lines. Leaves 10,096.
3. Format conversion.
4. `judge_pivots.py`: LLM-as-judge, N runs, majority vote, 6 failure-pattern tags. About 11K pivots, "~6% of turns". **This script is not in the repo.**
5. Train/validation split.
6. GRPO on single-turn local rollouts (agent `tool_simulation_agent`, `max_steps: 1`; `configs/swe_pivot.yaml:10-14`).

**Failure-pattern tags**, with prevalence (multi-label) (`README.md:205-229`):
- P1 consumer-vs-producer, i.e. editing the symptom site: 39%
- P2 backing off a correct fix: 14%
- P3 incomplete fix: 12%
- P4 over-engineering: 21%
- P5 hypothesis roulette: 68%
- P6 execution errors: 18%

**Selection rule** (`config-training-and-agent-ref.md:128-151`, `SKILL.md:95-110`):
- Prefer clean, positive source trajectories, and source tasks whose group had mixed rewards.
- Profile each candidate pivot with ≥8 on-policy local rollouts.
- Keep groups that contain both 0 and 1.
- Discard all-1 groups ("already easy") and all-0 groups ("too hard, underspecified, or mismatched to the verifier").
- When data is abundant, drop the easiest retained pivots first.
- If all source rewards are equal, say so in the metrics and rely on on-policy profiling.

**Verifier levels** (`app.py`):
- **L0 tool match** (`:295-306`). Exact name, or the same category, where `unknown` never matches by category. Categories (`:54-158`), each name also registered in CamelCase (`:57-61`):

  | Category | Names |
  |---|---|
  | edit | 18 names |
  | bash | 12 |
  | search | 17 |
  | read | 9 |
  | finish | finish, submit |
  | think | think |
  | planning | 12 |

- **L1 target match** (`:309-385`):
  - *Edit*: the editor sub-command must be present on both sides or neither, and equal. Paths are compared on their last 2 components. For `view`, if both sides give a range, overlap / expected_size must be ≥0.8; a missing range passes.
  - *Bash*: the command must be present on both or neither. The verb (first token, basename) must match. Paths extracted by regex must share at least one 3-component suffix. Whole-command SequenceMatcher must be ≥0.3 (`_BASH_TARGET_SIMILARITY_FLOOR`, `:392`).
  - *Search/read*: 3-component path suffix equality.
  - Anything else passes.
- **L2 similarity** (`:407-449`). Strings are capped at 2,000 characters (`:397-404`).
  - `str_replace`: `min(SM(old_str), SM(new_str))`. A side that is empty on one end only scores 0.
  - patch, diff or content: SM of the content.
  - bash: SM of the command.
  - Other categories: 1.0.
- **Reward** (`:681-729`):

  | Mode | Similarity range | Reward |
  |---|---|---|
  | binary | ≥ 0.8 | 1 |
  | binary | < 0.8 | 0 |
  | continuous | ≥ 0.8 | 1 |
  | continuous | 0.4 to 0.8 | `0.5 + 0.5·(s−0.4)/(0.8−0.4)` |
  | continuous | < 0.4 | 0 |

  Continuous mode then multiplies by `(1 − 0.1·min(len(edit_content)/500, 1))`.
- **Code vs README.** The code default is `reward_mode="binary"` (`:491`), while the README says continuous (`:248`). The README notes that "The PivotRL paper uses binary rewards for SWE-Bench" (`:103`).
- **Failure codes** (`:473-480`): `none`, `expected_action_invalid`, `model_output_invalid`, `tool_name_mismatch`, `target_mismatch`, `similarity_below_threshold`, `unknown_error`.

**Generic pivot comparator knobs** (`config-training-and-agent-ref.md:45-110`).
- `word_count_similarity_threshold`: the score is `intersection / (|expected| + |actual|)`, so its maximum is 0.5. Tune it on the pairs whose verdict flips, by offline re-scoring.
- Parallel calls: use `f1 = 2·matched/(expected+actual)`. It is the only mode that penalizes missing and surplus calls symmetrically.
- Nested JSON string arguments are only loosely checked; unwrap them.

**Row contract** (`SKILL.md:62-72`). One model call is one row. A call with no tool call is a `message`; one call is a `function_call`; two or more are a `function_call_batch`. Tool calls take precedence over narration.

**Segmentation pitfalls** (`rollout-artifact-pitfalls.md:9-80`).
- Reconstruct each model call from execution runs: a batch ends only at a tool result or a message. Reasoning items are transparent (agreement 76% vs 100%, as in TL;DR 17).
- Cross-check against the producer's own model-call count, and expect an off-by-one when the first turn is a seeded greeting.
- The pivot's own reasoning must be withheld from its prefix. Hoisted reasoning is a forward leak (44% of prefixes in one source). The repair keeps the first lead reasoning item and redistributes or drops the surplus.

### 2.4 ToolSandbox milestones and minefields (vendored, Apple ToolSandbox)

`NG/resources_servers/toolsandbox/tool_sandbox/common/evaluation.py`:
- **`SnapshotConstraint`** (`:726-757`): `(database_namespace, snapshot_constraint fn, reference_milestone_node_index?, target_dataframe?, column_similarity_measure?)`. Column measures default per namespace: exact, ROUGE-L, tool-trace exact, and so on (`:672-725`).
- **Constraint functions.**
  - `snapshot_similarity`: requires equal row count and a column superset. Builds a −log(col-sim) cost matrix and uses Hungarian assignment; returns `exp(−mean cost)`, i.e. a geometric mean (`:302-362`).
  - `addition_similarity`: the snapshot must contain the reference snapshot; the anti-join is then compared with the target (`:365-414`).
  - `removal_similarity`: the swap of `addition_similarity`.
  - `update_similarity`.
  - `guardrail_similarity`: exact equality with the reference (`:648-663`).
- **`Milestone`** (`:761-797`) holds a list of constraints. By default every non-sandbox DB with no constraint in this milestone, its reference or its predecessors gets an auto-added **guardrail** ("must not have changed"), relative to the predecessor and reference milestones (`_add_guardrail_constraints`, `:1019-1064`).
- **Milestone similarity** is the geometric mean of the constraint similarities. Guardrails multiply in but are excluded from the count, so passing guardrails do not inflate the score (`:868-937`).
- **`Minefield(Milestone)`** marks states that must never occur. Any non-zero minefield match nullifies the episode: `similarity = int(minefield_similarity == 0) · milestone_similarity` (`:800-810, 984-1006`).
- **Matching** (`MilestoneMatcher._dfs`, `:1079-1186`). Find the 1:1 mapping from milestone to snapshot index that maximizes the sum of similarities, subject to:
  - snapshot indices are non-decreasing in assignment order;
  - assignment order is a topological order of the DAG (edges default to a chain, `:1066-1077`).

  The search is DFS over zero-in-degree milestones. Candidate snapshots are pushed into a max-heap, iterating from the latest snapshot backwards so ties prefer later snapshots. A branch is pruned if `assigned + current + remaining_nodes − 1 ≤ best`. Similarities are cached per (snapshot, reference snapshot). The result is `mean over milestones`, and an empty DAG returns `None`, which is treated as 1 (`:1188-1236, 1249-1302`).
- **For M5/M6.** The matcher returns `milestone_mapping: {milestone_idx: (snapshot_idx, sim)}`, which is exactly "which step satisfied which milestone". Φ(s_t) can be computed as the mean similarity of milestones whose matched snapshot is ≤ t. Minefields act as a hard gate G, guardrails as per-milestone collateral checks, and ROUGE-L columns as soft matches. This is the only DAG-aware partial-progress implementation found in these sources.

### 2.5 Verifier fixtures, re-verification and ATIF

- **`verifier_fixture.py`**:
  - Cases are JSONL with `kind ∈ {full_reward, zero_reward, malformed, determinism}`. `malformed` requires `expected_error` and forbids `expected_reward` (`:21, 39-55`).
  - `exercise_verifier_fixture(reward_range, higher_is_better, determinism)`:
    - requires full/zero/malformed cases, plus determinism if seeded (`:190-195`);
    - asserts each reward lies in range;
    - full and zero cases must equal the range endpoints (`:223-234`);
    - determinism means a re-run on a *fresh server* returns the same value (`:235-238`);
    - `update_expected` rewrites the file atomically (`:142-160, 246-252`).
- **Reverify modes.** `ReverifyMode = stateless | unsupported | unknown` (`base_resources_server.py:75-85`). An `unknown` server needs `--force` and writes `unsafe_`-prefixed outputs (`config-training-and-agent-ref.md:112-119`). Stateful graders declare `UNSUPPORTED`: ViBench ("grading depends on live app and database state", `vibench/app.py:36-38`), `terminal_bench_4`, `genrm_compare`, `deepswe`.
- **ATIF re-verification** (`NG/nemo_gym/atif_reverification.py:15-120`). It projects a "deliberately bounded canonical ATIF v1.7 subset" into the Responses-shaped payload for *stateless* verifiers only. Each trajectory is joined to a materialized task by an explicit manifest `(trajectory_path, _ng_task_index, _ng_rollout_index, expected_sha256)` (`:101-110`). It raises `AtifProjectionError` instead of guessing ("cannot be projected without changing what is scored", `:88-89`).
- **For M5 replay.** NeMo Gym cannot re-verify stateful trajectories from transcripts. M5 replay (materializing state per step) is the missing piece. Port the strictness pattern: explicit join manifest, source hash, and refuse-to-guess.

### 2.6 SWE-bench grading and anti-cheat

- **Grading** (`NG/resources_servers/swebench/app.py:323-422`):
  - Spin up a *fresh* eval sandbox.
  - Extract the agent's patch with `git --no-pager diff` in the original sandbox, then stop that sandbox.
  - Run the upstream `run_instance` in the fresh container; `reward = int(resolved)`.
  - `evaluation_completed = completed and no sandbox error` is reported, but **not mapped to `mask_sample` here**. That gap is handled by the agent-side policies in §2.2.
- **Anti-cheat** (`anti_cheat.py:15-59`, `anti_cheat_setup.sh:1-25`). Run during `seed_session`, before the agent gets control:
  - point HEAD at an orphan-ish branch `_nel_work`;
  - delete `packed-refs`, `ORIG_HEAD`, `FETCH_HEAD`, all other refs, tags, remotes and logs;
  - `reflog expire --expire=now`, then `git gc --prune=now`;
  - log counts before and after.

  It is best effort: on failure the unscrubbed sandbox is handed over and a log line is written.

### 2.7 Token capture (training-time credit only)

- Loss masks follow provenance. Policy-generated tokens get mask 1 and keep their log-probs; prompt and interstitial tokens get 0 (`NG/nemo_gym/token_id_capture/builder.py:16-36, 46-70`).
- One rollout can yield several independent chains (sub-agents, forks).
- The build is masked when any of these holds (`consumer.py:180-203, 220-295`):
  - the terminal call is unattributed;
  - more than one root or chain exists under the strict policy;
  - a retry is unresolved;
  - the snapshot is empty or incomplete;
  - the delivery has no generated tokens.
- OSWorld exact traces keep "one logical rollout ... one reward and advantage" across compaction (`osworld_agent/README.md:57-64`).
- **For Crucible.** Per-turn spans exist implicitly (one output item per model call), but no per-turn reward channel exists.

### 2.8 Judges and partial credit elsewhere

- **GDPVal rubric scoring.**
  - A judge panel draws one member per trial, with a seeded draw for reproducibility, and routes audio/video tasks to capable judges (`NG/resources_servers/gdpval/judge_panel.py:14-70`).
  - The structured prompt asks for per-criterion `GRADE[x] out of MAX_POSSIBLE_POINTS[y]` and `FINAL_SCORE[x] out of MAX_POSSIBLE_SCORE[y]` (`scoring.py:136-160`).
  - The binary extractor "without inventing zeros for missing fields" averages `criteria_scores` (`:201-224`).
- **`generative_reward_model`** trains a GenRM policy against ground-truth score and ranking. It uses `parse_failure_penalty = -100.0` (`app.py:28-31`). It is a judge-training environment, not a step PRM.
- **Process reward models.** None were found in NeMo Gym. No step-level PRM server exists, and step rewards are summed (OpenEnv adapter, §1).

---

## 3. MiMo v2.6 RL data (`MD/`)

### 3.1 Layout and contract

- `code.parquet`: 2,698 rows. `instance_json` holds `{cwd, docker_image, instance_id, problem_statement, test_command, test_patch, verifier_timeout_sec}`; the reward is binary tests.
- `cyber.parquet`: 1,000 ARVO rows; the reward is rule checks.
- `general/envs/`: 267 directories, 266 with `verifier_meta.json`.
- `README.md:20-29` lists the domains.

**No milestone, checkpoint or progress constructs exist in any task.** Grep over metas and verifiers found only "partial" inside pass anchors.

Each general env has:
- `manifest.json`: workspace and system uploads, a sidecar with MCP servers on ports 39101-39105, and a `verifier` block writing to `/logs/verifier/reward.json` and `reward_detail.json` (`E/manifest.json:69-93`);
- `instruction.md`;
- `workspace/` with xlsx, pdf, docx and pptx inputs and deliverables;
- `system/<mcp>/{schema.sql,state.db}`;
- `tools/*.py`.

Top-level meta fields (counts across 266 envs):
- `input ∈ {both 107, db 93, workspace 65}`
- `act ∈ {none 201, mutate_db 60, both 3, edit_workspace 1}`
- `output = none`, `reward = rl_grade`

### 3.2 Rubric item schema (C4 candidate)

Example item (`E/verifier_meta.json:3-12`):
```json
{"id":"unadjusted_technology_change","tier":"critical","method":"llm",
 "question":"Does the final response identify ... YoY revenue change correctly?",
 "pass_anchor":"Ground truth from the controlled workbook row KF-TECH-GROWTH-001 ...: -660 bps, a 6.6% decline. Accept equivalent percentage or basis-point notation, but not a blended value or the per-billing-day measure.",
 "files":["answer.md"]}
```

| Field | Count |
|---|---|
| id, tier, method | 1,427 each |
| question | 1,247 |
| pass_anchor | 1,219 |
| files | 1,214 |
| fn (rule) | 215 |
| ground_truth (string or object) | 161 |
| weight | 123 (values 0.08 to 24; default 1) |
| description | 63 |
| dim | 7 |
| tolerance | 6 |

Rule item example: `{"id":"no-unrelated-mutation","tier":"sanity","method":"rule","fn":"check_no_unrelated_mutation","ground_truth":"All non-target rows ... must match the pinned pre-task database"}`.

- **Tier is not used in scoring.** The sidecar grader weights by `weight` only (`E/verify.py:395`). Tier is a label for authoring and analysis.
- **`pass_anchor` structure.** In practice it contains (i) the ground-truth source (a controlled row id or system record), (ii) the exact value, (iii) accepted equivalences ("Accept equivalent ... notation") and (iv) named wrong alternatives ("not a blended value or the per-billing-day measure"). This is what makes the binary decision reproducible.

### 3.3 Grading algorithm (`E/verify.py:343-405`, sidecar "scoring system v2")

1. **Source-conservation gate** (if `src_protect`). Any protected file modified or deleted, or any protected table's hash changed, fails the gate. New files created by the agent are not penalized (`E/_helpers.py:536-560`).
2. **Gate items** (`gate: true`, rule) run first. If any gate scores <0.999, `score = 0` and the LLM items are skipped.
3. **LLM items** (`method=llm`, text judge) are judged **one item per call**. Evidence is cached per file set (`:311-340`). Each item takes `votes` votes (default 1, env `GRADE_VOTES`), with up to 2 parse-aware retries per vote. The score is the median, clamped to [0,1]. No usable verdict raises `_JudgeUnavailable`.
4. **Rule items** always run, even after a gate failure, because they are cheap and useful for audit (`:386-389`). Each runs in a subprocess as `check_code + _helpers.py`, printing a `___R___` JSON marker, with a 300 s timeout. Any `random` import scores 0, since checks must be deterministic (`:15-48`).
5. **Aggregate.** `score = Σ_i w_i·s_i / Σ_i w_i` over non-gate atoms, excluding `agent` items and `judge=vision` items in the sidecar. Gates are not in the denominator. `passed = no gate failed and every item ≥0.999`. `by_method` gives per-method weighted means (`:390-405`).
6. **Judge unavailable.** Returns `{"score":0,"reward_error":"judge_crashed"}`, and `run_verify.py` then writes `reward.json` **without** a `reward` key (`E/run_verify.py:102-151`). The trainer side then masks with `missing_or_invalid_reward_json` (`VL/recipes/general/general_agent/environment.py:558-576`). A reward outside [0,1] is also masked (`:577-578`). An optional `reward_binary_fullscore` maps any score below 1 to 0 (`:581-583`).

Text judge prompt (Chinese in source, `E/verify.py:86-99`), translated:
> You are verifying whether a work deliverable satisfies an **objective checkpoint** (it has a unique right answer; it is not about aesthetics). Below are the extracted texts of the deliverable files and the checkpoint. Judge strictly: satisfied=1, not satisfied=0. **Only 0 or 1, no partial credit** (partially satisfied = not satisfied). **Judge only from the given text evidence**; if you cannot see evidence, judge 0; do not speculate. The evidence may contain [truncated…] markers or a "⚠ content not extracted" list: if the checkpoint's target content falls in the truncated/unextracted part, judge 0 and write "evidence truncated" or "evidence not extracted" in `why` (so audits can tell them apart; still do not guess a pass). … Output only one JSON object: `{"results": {"<item id>": {"score": 0, "why": "≤25-char basis"}}}`

Each item is rendered as `- [id] question (satisfied=1: pass_anchor)` (`:323-325`).

**Evidence extraction** (`E/_helpers.py:144-306`):
- Document order is preserved.
- Tables are rendered as grids with row and column labels.
- Content the extractor cannot render (images, charts, formats) is listed, so the judge can distinguish "absent" from "not shown".
- Truncation is structure-first and always marked; the default budget is 20,000 characters.
- If the target file is missing, `locate_deliverable` searches by keywords from the filename (`E/verify.py:284-306`).

**Judge transport** (`E/verify.py:64-185`):
- A fallback chain over `GA_JUDGE_KEY{,2,3}`, with a random starting point for load balancing.
- A shared phase budget of 600 s.
- Per-attempt read timeout of `min(500, left−20)`: "killing a queued request and retrying = losing queue position + double traffic".
- Up to 4 attempts with backoff from 5 to 20 s.
- Sticky failover after 2 failures (a hard 401/403/404/429/503 or timeout counts as 2).
- "A valid 0 is a verdict and does not trigger retry."

**Rule-check primitives** (`E/_helpers.py`):
- `db_query` is read-only, and checks are statically barred from `db_execute` (`:574-`).
- `deep_score(exp, got)` (`:415-488`) averages leaf checks:
  - numbers: tolerance `max(0.01, 0.5%·|e|)`;
  - scalar lists: multiset Jaccard;
  - object lists: aligned by the first field (which must be unique), recursing per item, scoring missing items 0, **and adding a 0 check for every surplus item**, so a lazy "copy the whole pool" answer lands near 0 while a near-miss keeps partial credit;
  - dicts: recursion per key.
- A mutation-task example (`s3k_0018`, `check_code`):
  - `check_pending_lots_flagged`: immutable business fields are unchanged; there is a new exception event with semantic code and note regexes, `by` and `idempotency_key`; the current fields equal the latest event.
  - `check_no_unrelated_mutation`: SHA-256 of `json.dumps(rows minus mutable columns, sort_keys)` equals a pinned digest.

---

## 4. XiaomiMiMo verl fork (`VL/`), mimoagent and uni_agent

### 4.1 Advantage pipeline order (`trainer_base.py:_compute_advantage`, 1917-2150)

1. Fetch `uid, response_mask, rm_scores, …`, plus `tool_call_error_mask`, `repetition_mask` and `turn_index` when those channels are enabled (`:1928-1940`). `token_level_scores = rm_scores`.
2. Apply the `mask` strategy for tool-error and repetition channels; repetition `early_stop` masks the prefix (`:1945-1984`).
3. KL in reward (off in MiMo), else `token_level_rewards = token_level_scores` (`:1988-1994`).
4. `reference_penalties.shape_training_rewards`: a group length penalty on the session's final scalar, re-placed at the last generated token (`:1996-2006`; `arvo_penalties.py:100-144`).
5. Web-dev group reward rewrite: a relative pick across siblings. A missing row is set to INVALID, meaning "replace with the group's mean valid reward" (`:2008-2009`; `VL/recipes/design/webdev/group_reward.py:47-52`).
6. Rollout-correction IS weights (`:2011-2021`).
7. **Infra exclusion.** If `DROP_INFRA_FROM_GROUP=1` or reference penalties are on, every `is_infra` row's uid is renamed to `__infra_excluded__<uuid4>`, making it a singleton group with advantage 0 (`:2023-2040`).
8. Optional `group_advantage_by_harness`: uid becomes `uid::agent_type` (`:2046-2052`; `utils.py:262-266`).
9. **`compute_advantage_for_multi_trajectories`.** For GRPO: select the final output (max output index) per `{uid}_{session_id}`, run GRPO on those rows, then scatter `scalar × response_mask` to every row of the session (`utils.py:204-277`). Other estimators (GAE, RLOO, OPO, GDPO, …) fall through to the stock `compute_advantage`.
10. `deep_failure_mask` (experimental) (`:2065-2101`).
11. `adv_reduction` / `adv_set` marked-token edits (`:2102-2108`).
12. `adv_signed` channels, max over κ, with `row_weights` = prompt-mean loss weights (`:2110-2137`).
13. `reference_penalties.apply_tool_penalty` (`:2139-2150`; broken call, TL;DR 2).

### 4.2 Estimators

- **GRPO** (`core_algos.py:267-353`):
  - `scores = Σ_tokens token_level_rewards`.
  - If `invalid_reward_value` is set (MiMo uses -999; config `ppo_trainer.yaml:133-139`), invalid scores are replaced by the mean of the valid group members (0 if there are none), and their advantage is forced to 0 after normalization (`:306-326, 348-349`).
  - Normal case: `A = (s − μ_g)/(σ_g + 1e-6)`, or `s − μ_g` when `norm_adv_by_std_in_grpo=false`. A singleton group has μ=0 and σ=1.
- **GDPO** (`core_algos.py:383-490`): `A = whiten(Σ_k w_k · GRPO(r_k))`. Requires `algorithm.gdpo_reward_keys` present in `non_tensor_batch`, with optional `gdpo_reward_weights`. Each component is placed at the last valid response token.
- Also registered: `grpo_passk`, RLOO, OPO, REINFORCE++, ReMax, GPG, and the optimal-token-baseline variants (`core_algos.py:88-110`).

### 4.3 Turn-span operators (exact math)

- **Spans.** Each model call appends `(turn_start, len(response_mask))` *after* `append_generated`. Spans are therefore in response coordinates and cover only generated tokens (`VL/recipes/arvo/agent_loop.py:245-247`). `trajectory_metadata` ships `llm_turn_spans`, `tool_call_error_flags` and `length_signals {prompt_length, response_length, decode_length, tool_length, prefill_length, turn_count}` (`recipes/arvo/trajectory_metadata.py:22-40`).
- **Tool-error flags.** A turn is flagged if any of its tool calls raised `FormatError` (`MA/agents/default.py:124-133`). It is a per-turn bool.
- **`tool_error_hits`** (`arvo_penalties.py:146-172`):
  - Skip `is_infra` rows.
  - Require `len(spans)==len(flags)`.
  - Require spans to be monotone and non-overlapping, with `coverage == generation_mask` exactly, else raise.
  - Set `hit[row, start:end] = κ` for flagged turns.
- **uni_agent equivalents** (`UA/framework/framework.py:479-530`):
  - `_turn_index` sets token→turn i inside span i and −1 elsewhere. On any inconsistency it **fails closed**: all −1, status `invalid_span`.
  - `_tool_call_error_mask` "Refuse[s] to guess when their lengths disagree: a zero mask plus an explicit alignment status".
- **`signed_rebalance`** (`signed_rebalance.py:101-172`). Let the row sign be `sign(median(A over valid tokens))` (`:195-200`), `w` the prompt-mean weights, and `m = w·A`.
  - Mass sums:
    - `removed = Σ m` over positive-row hit tokens
    - `pos_base = Σ m` over positive-row clean tokens
    - `added = Σ (κ−1)·|m|` over negative-row hit tokens
    - `neg_base = Σ |m|` over negative-row clean tokens
  - Edits:

    | Tokens | New advantage |
    |---|---|
    | positive row, hit | 0 |
    | positive row, clean | `A·α`, with `α = 1 + removed/pos_base` clamped to ≤ `max_scale` (2.0) |
    | negative row, hit | `A·κ` |
    | negative row, clean | `A·β`, with `β = 1 − added/neg_base` clamped to ≥ `min_scale` (0.5) |

  - Both signs conserve batch mass unless clamped. Conservation is asserted, and a warning is issued on clamp (`:46-98, 150-157`).
- **Reference profile** (`VL/recipes/arvo/REFERENCE_PENALTIES.json`):
  - `selected_rules = [tool_call_error (adv_signed, segment level, κ=2, min 0.5, max 2), length_penalty]`
  - `excluded_reference_rules = [tool_name_invalid, toxic_reasoning, overlong, agent_context_filter]`
  - It requires GRPO with Dr.GRPO, no KL, `filter_groups` on reward, and `loss_agg_mode=prompt-mean` (`arvo_penalties.py:74-98`).
- **Length penalty** (`VL/verl/utils/length_penalty.py:14-40`). Only *passed* rollouts (reward ≥0.5) are penalized, and only when the group pass rate is >0.5.
  - `excess_m = max(0, (v − anchor)/anchor)`, where the anchor is the 0.3-quantile of passed rollouts' {turns, input tokens, output tokens}.
  - The per-metric excesses are combined with max.
  - Penalty = `0.2 · t^1.5`, with t the normalized position in the [0, 1] excess band. It is subtracted from the scalar.
- **`deep_failure_mask`** (`trainer_base.py:204-256`; config `algorithm.py:140-167`). Per group, `k = ceil(α·median(turns of successful rollouts))`. It masks tokens with `turn_index ≥ k` on negative-advantage rows (`mask_failure`) or on all rows (`mask_both`). The config marks it EXPERIMENTAL and "a biased estimator (removes negative gradient mass) used only to test where deep-turn entropy growth comes from". `_depth_bucket_mass_metrics` logs signed advantage mass per turn-depth bucket (1-5, 6-20, 21-50, 51-100, 101+) under prompt-mean weights (`:166-201`).
- **Marked-token strategies** (`algorithm.py:68-105`): `monitor | mask | adv_reduction (A −= v) | adv_set (A = v) | adv_signed (κ ≥ 1)`. Production code-recipe defaults are all `monitor`/disabled (`VL/recipes/code/config/train.yaml:164-176`).

### 4.4 Infra masking and group filtering

- Infra categories (`recipes/arvo/agent_loop.py:76-84`, `recipes/general/agent_loop.py:78-87`): `setup/failed`, `rollout/pod_conn_timeout`, `rollout/seq_timeout`, `reward/exception`, `reward/env_error`, and `reward/testbed_corrupted` (general only).
- `extra_fields["is_infra"]` is always present. `invalid_reward_for_infra` controls whether infra rows score -999 or 0.0 (`agent_loop.py:625-728`).
- `_failure_output` records production crash lessons. Failed rollouts must still provide:
  - a float `reward_score`, or the whole batch loses `rm_scores`;
  - non-None log-probs, or the column is dropped when the failure lands at index 0;
  - a non-empty prompt, since an empty prompt asserted in padding and "one pod that never reached 'environment ready' out of 32 trajectories was enough to kill a 64-GPU step";
  - `min/max_global_steps`.

  It emits one EOS token with mask 1, so that an all-failure batch does not divide by zero.
- **DAPO filter on the non-infra subset** (`replay_buffer.py:257-269`):
  - no valid rows: evict at 0;
  - one valid row or zero std: evict;
  - otherwise keep.

  Tests document why: "10 solved + 6 infra zeros" would otherwise pass the filter and then carry no gradient (`VL/tests/trainer/ppo/v1/test_infra_group_filtering_on_cpu.py:14-50`).
- **Reward-time infra** (mimoagent). `REWARD_TESTBED_CORRUPTED = "reward/testbed_corrupted"` is returned when, for example, the gold `test_patch` fails to apply because an agent-planted file collides with it. Trainers mask the whole sequence (`MA/environments/datasets/base.py:29-35`). When verifier and judge both run, a judge failure "fails closed: reward 0 with the category kept" (`:275-325`).

### 4.5 mimoagent rubric judge (`MA/environments/rubric_judge.py`)

- **Instance schema.** `rubric.rubrics` is a list of strings or `{text, category?, subcategory?, how_to_judge?, id|label?}`. Category is an open vocabulary (for example `final_state`, `process`, `user_communication`) and "a pointer to the primary evidence, not a restriction". Malformed items raise: "a silently dropped criterion would be a wrong reward" (`:148-202`). An optional graded `rubric.solution.levels` is listed best-first, each `{score∈[0,1], description, how_to_judge?}`, with scores non-increasing; plus an optional `rubric.solution_weight` (`:205-256`).
- **Aggregation** (`:259-315`). `B = mean(verdict==1)`, where any non-1 value, including missing, counts as 0. `S` = the reported level's score, or 0 if invalid. `rubric_reward` = `w·S + (1−w)·B` (default w=0.5) or `S·B` (the `product` mode, "used by the RL side to rank test-passing rollouts"). It is pure and can re-score offline.
- **Combine with the programmatic verifier** (`:129-145, 377-386`):
  - `programmatic` (default: "adding rubrics to a test-graded dataset never changes what it trains on unless asked to")
  - `rubric`
  - `product` ("tests gate, rubrics scale")
  - `weighted` (`(1−w)·prog + w·rubric`, default w=0.5)

  "Rubrics never gate each other; the only gate is the dataset's own verifier."
- **Judge agent.** A `DefaultAgent` with its own model, `bash` + `read` tools, and `step_limit 100`. It runs in the *doer's pod* with workspace `/tmp/.mimo-rubric-judge/{task.md, rubrics.json, solution.json?, response.md, model_patch.diff, trajectory}` (`:318-349, 458-535`).

  System prompt highlights (`:92-116`), quoted:
  - "Judge the work AS DELIVERED. Never fix, complete, or improve the doer's work before judging … If a check requires mutating state …, keep the mutation reversible and restore the doer's state afterwards."
  - "When a rubric concerns the behavior of the delivered system, verify in the environment rather than trusting what the response or trajectory asserts: read the actual code/diff, run things."
  - "When the primary evidence … is unavailable … judge from the best remaining evidence — in both directions … State the limitation in your `reason`."
  - "Be strict: a rubric is 1 only if the evidence establishes it is satisfied. Partial credit is 0. Every rubric applies to every rollout: a rubric that names an obligation the doer never carried out … is 0."

  Output is `verdicts.json` = `{"solution":{"level","reason"}?, "verdicts":[{"id","verdict":0|1,"reason"}]}`. It is read once, framed by random begin/end markers to survive shell banners. A missing or invalid file returns `error_category=REWARD_TESTBED_CORRUPTED` and is masked (`:422-454, 579-611`).

### 4.6 Anti-hack

- **Runtime guard** (`MA/agents/antihack.py`). An action interceptor checks bash fields (`Bash.command`, `exec_command.cmd`, `apply_patch.input|patch`) and path fields (Read/Edit/Write/Grep/Glob in 3 tool catalogues) (`:65-84`). It matches recall-first regexes (`:90-195`):
  - git network and ref archaeology (`fetch|pull|clone|remote|ls-remote`, `reflog|stash|bisect|show-ref|for-each-ref`, `branch -a/-r`, `tag`, `--all|origin/|upstream/|refs/|FETCH_HEAD`, `cat-file|fsck|…`);
  - raw GitHub, codeload and API commit, pull or release URLs;
  - unpinned or VCS installs (`pip install git+|https://|-U`, `npm … github:|latest`, `go get @latest`, …);
  - eval artifacts (`.eval/`, `secret_cases`, `hidden_tests`, `private_tests`, `answer key`, `golden_patch`, `test_patch`, `ground truth`).

  A hit returns a dummy `"Permission denied. Resource unavailable."` with empty metadata, so the policy cannot tell that a guard fired (`:246-343`). The block is telemetry only in v1: "the seam to a reward/penalty channel is here" (`:267-269`). The stage-2 LLM intent judge is a stub.
- **Environment hygiene** (`MA/environments/datasets/base.py`):
  - `git_leak_prevention ∈ {hide, strip (default; falls back to hide), none}` (`:46-55`).
  - `anti_hack_cleanup` purges build residue: `/tmp/*.log`, `test_files.json`, jest/pytest/go-build caches, `/tests /logs`, DB logs and `task_description.md` (`:593-640`).
- **Test-patch hygiene** (`MA/environments/datasets/opensource_code.py:1-40, 211-236`).
  - The verifier script ships inside `test_patch`, applied only at reward time, so it never exists while the agent runs.
  - Reward order: reset every path the patch touches to the base ref, then `git apply`, then `test_command`, with reward 1 iff exit 0.
  - A reset or apply failure produces `testbed_corrupted` and is masked.

### 4.7 Multi-harness training

- **Harness spec.** `MIXED_HARNESS_SPEC` lists `{label, config}` arms (`VL/config/agent/code/mix-four-whitebox.yaml`).
- **Arm selection** is deterministic (`mimoagent_runner.py:109-137`):

  | Mode | Arm index | Effect |
  |---|---|---|
  | `prompt` (default) | `seed + dataset_index` | whole group on one harness |
  | `paired_validation` / `session` | `rollout_index` | harness varies within the group |
  | `step-hash` | `sha256(seed, harness_round, instance_id)` | per-step reshuffle |

  "There is no single-harness branch … a one-armed run is a spec with one entry."
- **Per-harness grouping.** `group_advantage_by_harness` requires `agent_type` on every trajectory tag (`trainer_base.py:2046-2052`). Per-harness raw pass-rate metrics are logged (`utils.py:173-193`).
- **Trajectory selection.** uni_agent `trajectory_selection ∈ {all, longest}`; code uses `longest` (`UA/framework/framework.py:58-71`; `train.yaml:81`). `mask_unfinished_episode` defaults to false (`train.yaml:66`).

---

## 5. Synthesis for the focus questions

### 5a. Injecting A_t = A^GRPO + η·(normalized shaping return-to-go) into verl

**What verl provides:**
- per-session scalar GRPO broadcast (`utils.py:204-277`);
- span metadata in `extra_fields` (`llm_turn_spans`) or `turn_index` (uni_agent);
- the infra uid-exclusion trick;
- prompt-mean row weights;
- the mass-conservation machinery and metric conventions (`signed_rebalance.py`).

**What it lacks:** an additive per-span operator.

**Recommended operator** `apply_turn_shaping`. Insert it after `trainer_base.py:2137` (after the signed channels, so that `rebalance_dense`'s median-sign rule sees the outcome sign) and before reference penalties.

Inputs per row: `llm_turn_spans`, `turn_shaping` (one float per span, written by the Crucible agent loop from `steps.jsonl`), and `is_infra`. Validate as in `arvo_penalties.py:157-171`. On any misalignment, fail closed for that row (shaping 0, status metric), as uni_agent does.

Let trajectory i in group g have turns t = 0..T_i−1, potential Φ_{i,t} at the start of turn t, and Φ_{i,T_i} at the end.

- *Invariant form* (DESIGN P: Φ(terminal)=0; set `Φ_{i,T_i} := 0`): the shaping return-to-go is `G_{i,t} = Σ_{u≥t}(Φ_{i,u+1} − Φ_{i,u}) = −Φ_{i,t}`.
  - This is a **state-only baseline**. It is unbiased for any Φ, and it reduces variance only insofar as Φ tracks V. A_t = R_i − μ_g − η·Φ̂_{i,t} is the actor-critic advantage with V ≈ μ_g + ηΦ̂.
  - It gives *no* credit for raising Φ. That credit comes from R itself.
  - Under this form the M6 validity gate ("Φ at midpoint predicts outcome, AUC ≥0.75") is exactly the condition for the term to help. (This is analysis, not from the sources.)
- *TD form* (C3-like): `δ_{i,t} = Φ_{i,t+1} − Φ_{i,t}` gives each action credit for its own progress. It is biased unless Φ = V, so use it only with C3-calibrated Φ (VinePPO's `V̂(s_{t+1}) − V̂(s_t)`), and bound η.

Normalization and mass:
- Pool all (i,t) entries in group g, excluding infra rows. Weight each turn's entry by its prompt-mean token weight `w_{i,t} = n_tokens(i,t)·w_row(i)`.
- Center: `Ĝ_{i,t} = (G_{i,t} − μ^w_g)/(σ^w_g + ε)`. Weighted centering guarantees `Σ w·Ĝ = 0` per group, so no net mass is injected (the `adv_set` lesson).
- Scale by the outcome scale: `A_{i,tokens∈t} += η·σ_g(R)·Ĝ_{i,t}`. Under Dr.GRPO, A^GRPO is in reward units.
- Set `returns = advantages`.
- Log the `adv_pos/neg_sum` pre and post, and the depth-bucket mass (`_depth_bucket_mass_metrics`), to catch deep-turn drift.

Interaction with filtering: put the gated scalar R (with the κ(1−O)Φ term) in `rm_scores`. Otherwise `filter_groups.metric=reward` evicts all-fail groups (`replay_buffer.py:257-269`) before shaping can act.

Multi-row sessions (context compaction): spans are per row, and the scalar is broadcast per session. Index turns per session, not per row, and ship the `session_id` already present in keys (`utils.py:236-243`).

Alternative, with no code change: use GDPO with `gdpo_reward_keys = [O, p, e, Φ_end]` for episode-level multi-channel credit (`core_algos.py:383-490`).

### 5b. Decision points as input to M5 stages and M6 credit

- **Detector** (from PivotRL practice):
  1. Choose candidate steps on successful recorded traces.
  2. Branch K≥8 continuations from the restored snapshot (Crucible C3).
  3. Keep steps with mixed outcomes, drop 0/K and K/K, and rank by closeness to 0.5 or by variance.

  The ~6% pivot rate is a sanity prior. An LLM judge with N-vote majority and P1-P6 tags is a cheaper pre-filter, but its script is not available, so its prompt must be authored. P1-P6 are a ready taxonomy for M4 signal tags.
- **Stage boundaries.** Use `swe_pivot`'s category map (search/read → edit → bash/test → finish; planning; think) as the base features for stage induction. A category transition is a candidate boundary. Pivots (high-variance steps) mark where stage credit concentrates.
- **Tracker deviation.** Compare the policy's action at a recorded pivot with the expert's using the L0-L2 comparator (binary at 0.8). A deviation at a pivot is the signal to flag, not a deviation anywhere.

### 5c. Rubric items and judge prompts for C4

Adopt the union schema:
`{id, tier: critical|important|sanity, method: rule|llm|agent, judge: text|vision, category: final_state|process|user_communication, question, pass_anchor, how_to_judge?, files[], fn?, ground_truth?, weight?, gate?, dim?, tolerance?}`, plus optional graded `solution.levels`.

Use the MiMo text prompt (binary, evidence-only, truncation → 0 with reason) for checkpoint-time items, with one call per item, median of `votes`, and 2 parse retries. Use the mimoagent agentic judge for end-of-episode process and final-state items.

For the C4 lifecycle (candidate → validated → weighted), MiMo's per-item `weight` fields provide the slot. Their weights are hand-set; time-split AUC is not done anywhere in these sources. Keep tier as a prior only. Rule items run every step (cheap); text-LLM items run at checkpoints; agentic items run at the end, matching the tracker's cost tiers.

### 5d. What dense or process rewards did, and anti-hack lessons

Measured or documented:
- Whole-turn `adv_set −1` flattened the policy and lengthened outputs; it was replaced by sign-aware mass conservation (`signed_rebalance.py:27-29`).
- A -999 in GRPO groups collapsed the pass/fail advantage gap from 2.15 to 0.004 (`test_bridge_contract_on_cpu.py:285-286`).
- The reference profile keeps only tool-format-error and length penalties, and excludes 4 others (`REFERENCE_PENALTIES.json`).
- The deep-turn mask exists to investigate "deep-turn entropy growth" and is flagged as biased (`algorithm.py:143-156`).
- All production code-recipe penalties default to `monitor` (`train.yaml:164-176`).
- PivotRL prefers binary rewards for SWE (`swe_pivot/README.md:103`).
- MiMo general offers `reward_binary_fullscore` (`environment.py:581-583`).

No numbers on whether rubric partial credit helped were found.

Anti-hack: TL;DR 14 and §4.6. In addition:
- Efficiency penalties apply only to *passed* rollouts (`length_penalty.py:32`). Apply efficiency terms only inside the success gate, which is DESIGN's G·O·(1+βe).
- Rule checks must be deterministic: `random` is banned (`E/verify.py:19-20`).

### 5e. Masking conventions to standardize in C6

1. One record: `{mask, failure_kind (NeMo vocabulary + crucible:<kind>), failure_stage, failure_reason}`. A masked row still has a finite reward (TE-6).
2. In GRPO, masked rows leave the group statistics (uid renaming, or `invalid_reward_value` with mean imputation and advantage 0). Never use a raw -999.
3. The dynamic-sampling filter judges only unmasked rows; a single survivor means no signal.
4. Transport failure of a judge or verifier is masked. Unparseable output is masked after N retries, with a rate alarm. A grader *configuration* error raises.
5. Policy-attributable endings (horizon, explicit FAIL, unparseable action) are *not* masked. "Resolved but never submitted" (truncation) is masked. Pick the `anyswe` rule over `swe_agents`.
6. A reward file with no `reward` key, an out-of-range reward, or a failed test-patch apply is `testbed_corrupted`, and masked.
7. `kill_shaped` failures leave no row, and resume re-dispatches them.
8. Metrics publish `masked_pct`, `reward_unmasked` and per-agent coverage.

---

## 6. Mechanism table

| Mechanism | Source path:line | Use in Crucible | DEPEND / PORT / PATTERN |
|---|---|---|---|
| `mask_sample` + `failure_kind` + `failure_reason` on verify response | NG/nemo_gym/base_resources_server.py:107-148 | C6 record schema | PORT |
| Failure vocabulary + stages | NG/nemo_gym/failure_kinds.py:49-165 | C6 kinds; `crucible:<kind>` namespacing | DEPEND (stdlib-only module) |
| Failures sidecar; kill_shaped = no row; attempt cap 3 | NG/nemo_gym/rollout_collection.py:163-217, 1243-1284 | harness rollout runner (C3, training) | PATTERN |
| `judge_failsafe` (transport → mask; unparseable → score) | NG/nemo_gym/judge.py:15-121 | C4/C5 judge wrapper | PORT (amend unparseable rule) |
| `select_measured` / coverage metrics | NG/nemo_gym/reward_profile.py:842-950 | validity reports, calibration | PATTERN |
| `reward_components` (multi-key) | NG/nemo_gym/base_resources_server.py:151-164; example_tool_call_multireward/app.py:14-163 | `compose.py` emits components + scalar | PATTERN |
| Grader config error raises, never zeroes | NG/resources_servers/vibench/app.py:606-609 | all graders | PATTERN |
| swe_pivot L0-L2 comparator + thresholds | NG/resources_servers/swe_pivot/app.py:54-465, 681-729 | M5 trace alignment, tracker deviation, imitation-at-pivot | PORT |
| Pivot selection (≥8 rollouts, mixed-only) | NG/.agents/skills/nemo-gym-pivot-datasets/references/config-training-and-agent-ref.md:128-151 | decision-point detector via C3 | PATTERN |
| P1-P6 pivot failure taxonomy | NG/resources_servers/swe_pivot/README.md:201-229 | M4 signal tags, M5 stage labels | PATTERN |
| One-call-one-step segmentation; reasoning transparent; leakage repair | NG/.agents/skills/nemo-gym-pivot-datasets/references/rollout-artifact-pitfalls.md:9-80 | M5 segmentation + audit | PORT |
| F1 for multi-call; word-overlap ceiling 0.5 | config-training-and-agent-ref.md:62-110 | C4 rule items, R5 tests | PATTERN |
| Milestone/minefield DAG, guardrails, geometric mean, DFS matcher | NG/resources_servers/toolsandbox/tool_sandbox/common/evaluation.py:302-1302 | M5 milestone mapping; Φ; tracker; gate G | PORT |
| Verifier fixture kinds (full/zero/malformed/determinism) | NG/nemo_gym/verifier_fixture.py:21-253 | R1/R2 + determinism tests | DEPEND or PORT |
| ReverifyMode; ATIF strict projection | NG/nemo_gym/base_resources_server.py:75-85; atif_reverification.py:15-120 | M5 replay manifest (hash, explicit join) | PATTERN |
| SWE fresh-container grading; git history scrub | NG/resources_servers/swebench/app.py:323-422; anti_cheat_setup.sh:1-25 | coding Worlds | PORT |
| Agent mask policies (anyswe / swe_agents / osworld) | NG/responses_api_agents/anyswe_agent/app.py:103-113; swe_agents/app.py:3895-3915; osworld_agent/rollout_outcome.py:74-125 | C6 truncation rules | PATTERN |
| Token capture: loss mask by provenance; mask on gaps | NG/nemo_gym/token_id_capture/builder.py:16-70; consumer.py:180-295 | harness token capture for training | PATTERN |
| OpenEnv adapter sums step rewards | NG/resources_servers/openenv/app.py:227-243 | warning: do not collapse `steps.jsonl` | PATTERN |
| Harness evidence TE-1..TE-9 | NG/nemo_gym/harness_capabilities/README.md | harness admission gate | PATTERN |
| verifier_meta item schema | MD/general/envs/*/verifier_meta.json (E/verifier_meta.json:1-79) | C4 item schema | PORT |
| Gate → atoms → weighted mean; by_method | E/verify.py:343-405 | `compose.py` rubric part | PORT |
| Strict binary evidence-only text judge prompt | E/verify.py:86-99, 311-340 | C4 checkpoint judge | PORT |
| Judge fallback chain, phase budget, no-kill-on-queue | E/verify.py:64-185 | C4/C5 judge client | PATTERN |
| No `reward` key → mask | E/run_verify.py:102-151; VL/recipes/general/general_agent/environment.py:558-583 | grader output contract | PORT |
| `extract_evidence` with truncation + loss markers | E/_helpers.py:144-306 | judge evidence builder | PORT |
| `deep_score` with superset penalty | E/_helpers.py:415-488 | C4 rule items; R5 | PORT |
| `protect_diff`; no-unrelated-mutation digest | E/_helpers.py:536-560; s3k_0018/verifier_meta.json check_code | per-milestone collateral check (R6) | PORT |
| Post-GRPO operator slot; `returns` sync | VL/verl/trainer/ppo/v1/trainer_base.py:2053-2150 | A_t injection point | DEPEND (fork) |
| Session-final GRPO broadcast | VL/verl/trainer/ppo/v1/utils.py:204-277 | per-session indexing of turns | DEPEND |
| `llm_turn_spans` + span validation | VL/recipes/arvo/agent_loop.py:245-247; arvo_penalties.py:146-172 | `turn_shaping` alignment | PORT |
| `_turn_index` fail-closed | UA/framework/framework.py:479-530 | same, for uni_agent harnesses | DEPEND |
| `signed_rebalance` mass conservation | VL/verl/trainer/ppo/signed_rebalance.py:46-215 | zero-mass shaping; tool-error channel | DEPEND |
| GRPO `invalid_reward_value` mean-imputation | VL/verl/trainer/ppo/core_algos.py:306-351 | C6 in trainer | DEPEND |
| Infra uid exclusion | VL/verl/trainer/ppo/v1/trainer_base.py:2023-2040 | C6 | DEPEND |
| DAPO classify on non-infra subset | VL/verl/trainer/ppo/v1/replay_buffer.py:257-269 | dynamic sampling | DEPEND |
| GDPO | VL/verl/trainer/ppo/core_algos.py:383-490 | multi-component episode credit | DEPEND |
| Group length penalty (passed only) | VL/verl/utils/length_penalty.py:14-200 | e-term inside the success gate | PATTERN |
| deep_failure_mask + depth mass metrics | VL/verl/trainer/ppo/v1/trainer_base.py:166-256 | diagnostics for deep-turn drift | PATTERN |
| `_failure_output` well-formed failure rows | VL/recipes/arvo/agent_loop.py:625-728 | Crucible verl agent loop | PORT |
| Rubric judge (agentic, file verdicts, combine modes) | MA/environments/rubric_judge.py:92-454 | C4 end-of-episode judge; compose modes | PORT |
| `REWARD_TESTBED_CORRUPTED`; judge-fail fails closed | MA/environments/datasets/base.py:29-35, 275-325 | C6 | PATTERN |
| Anti-hack guard (dummy observation, patterns) | MA/agents/antihack.py:65-343 | World action filter; R-tests | PORT |
| Build residue purge; git leak prevention; test-patch reset | MA/environments/datasets/base.py:46-55, 593-640; opensource_code.py:1-40 | coding World setup + grader | PORT |
| Harness mixing modes; per-harness grouping | VL/recipes/code/mimoagent_runner.py:109-137; trainer_base.py:2046-2052 | multi-harness rollouts / C3 calibration | PATTERN |

---

## 7. Open questions and risks

1. **The `arvo_penalties.apply_tool_penalty` arity bug** (TL;DR 2). Do the MiMo results rely on this path, or on the generic `tool_call_error_penalty.strategy=adv_signed` path (`trainer_base.py:2110-2137`), which is correct? If Crucible depends on the fork, pin a commit and add a CPU test that calls `apply_tool_penalty`.
2. **The invariant-form A_t only reduces variance.** If M6 needs credit for *making progress* (the TD form), η must be validated against the outcome objective: an ablation of outcome success at fixed compute. No source supplies evidence that it helps.
3. **Group normalization of per-turn entries.** Turns do not align across trajectories. Pooling over (i,t) with token weights is our choice. An alternative is a per-turn-depth baseline, but depth buckets were only used for diagnostics in the fork.
4. **Dr.GRPO vs std-normalized shaping scale** (TL;DR 6). It needs a one-time η sweep per task family. DESIGN already states η is tuned per family.
5. **Unparseable judge output.** NeMo scores it, MiMo masks it. Choosing "mask" opens a hack: a policy whose outputs break the judge escapes negative reward. Mitigate with a rate alarm and a cap, for example masking at most X% of a group.
6. **Truncation masking.** `swe_agents` masks every max-iteration rollout, `anyswe` only resolved-but-truncated ones, OSWorld none. Masking all truncations hides "agent never finishes", which is a real policy failure. Recommendation: mask only resolved-but-unsubmitted.
7. **No time-split validation of rubric weights** exists in these sources. MiMo weights are hand-set, and tier is unused by the grader. C4's AUC-weighted lifecycle is novel; budget for the judge-variance study (`votes`, median).
8. **The ToolSandbox DFS matcher is exponential in the worst case** ("brute force … with pruning", `evaluation.py:1199-1206`). Offline mining over long coding traces with many milestones may need a chain-DAG DP (O(M·T)) instead.
9. **Missing pivot judge prompt.** `judge_pivots.py` and `terminal_pivot` are referenced but absent from NeMo Gym, so the LLM pre-filter prompt for decision points must be written by us.
10. **`filter_groups` vs dense credit** (TL;DR 5). Confirm with a dry run that all-fail groups with differing Φ survive the replay buffer when R includes κ(1−O)Φ.
11. **Anti-hack regexes are coding-specific.** Enterprise Worlds need equivalents for reading the grader's DB tables or protected files. MiMo's `src_protect` and the DB digests are the starting point, but no shipped env enables `src_protect`.
