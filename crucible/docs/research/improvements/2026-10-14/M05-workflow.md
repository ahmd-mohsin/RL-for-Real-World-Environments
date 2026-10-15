# M5 workflow: audit against the lab RL repos (2026-10-14, revised)

Crucible HEAD `1f89e2a`. Method: the internal audit plan (`00-plan.md`, kept local and gitignored, not in the repo). Every claim about Crucible carries a `path:line` under
`crucible/src/crucible/` (or `packs/…`); every claim about a reference repo carries `path@commit` under
`RL-Environments-Frameworks/<owner>-<repo>/`. Findings marked **[ran]** were reproduced by a probe; the probes live in
the session scratchpad (`scratchpad/m5/`, `scratchpad/m5v/`) and are described where they are used.

**Revision note.** The first version of this report listed 13 recommendations. Each one was re-tested against one bar:
it changes something that is wrong in what Crucible produces **today**, its benefit is measured, and applying it cannot
break a later stage. One recommendation passes. The other twelve are recorded in §7 with the reason each one failed the
bar; four of them rested on premises that the re-test disproved.

---

## 0. Summary

- **Where M5 matters.** Live RL episodes do not use M5's mined workflow model: their per-step Φ comes from the task's
  own grader subgoals (`verify/progress.py`, attached at `runtime/harness/native.py:319`). M5's model reaches a
  customer through one product path: `crucible label`, where the pack's `credit` hook turns it into per-turn weights
  and partial credit in the reward of SFT / KTO / DPO data (§1).
- **The defect.** On that path, Φ is written into the datasets without any check that it carries information about
  success. DESIGN requires that check (filter (d), §6.5), and Crucible already has the code for it
  (`credit/validity.py`), but no milestone or Φ channel goes through it. On the toy pack, two of the three mined
  milestones are present in every ticket, good or bad, and Φ in the middle of a ticket predicts the outcome at or
  below chance (AUC 0.40 on the tickets it was mined from, 0.50 on held-out tickets; corrected 2026-10-15 from 0.58, an older figure: the stored toy report and the built gate both give 0.50) **[ran]**. As a result the KTO
  file gives a wrong answer from a failed ticket the highest weight in the file, and two failed tickets with wrong
  answers earn partial credit **[ran]**. On coding, Φ at the midpoint is at chance (AUC 0.50), and the end-state Φ is
  placed on the last turn as if that turn had done all the work **[ran]**.
- **The recommendation (§7, R1).** Φ reaches a dataset only through a held-out validity verdict, one per use: a step
  channel (per-turn weights) and a terminal channel (partial credit). A channel that fails falls back to the encoding
  Crucible already has for "no dense credit". Applied to the toy, the datasets keep exactly the same rows; only the
  uninformative numbers go **[ran]**. Coding keeps its end-state partial credit, which passes (AUC 0.66, CI 0.53–0.80).
- **What the references contribute.** No reference repo validates progress checkpoints against outcomes (checked
  across all 31). The fix uses Crucible's own design rule and code. The references are ahead on mechanisms around
  milestones (guardrail roles, id-free row matching, per-step state capture, verified re-execution). None of them
  fixes a defect that is live today, so none is recommended now (§5, §7).

---

## 1. What the module does

**In plain words.** M5 takes recorded work (an agent session, a pull request, a ticket's event log) and builds a model
of the workflow. It splits the record into steps, rebuilds what the world looked like after each step, finds
**milestones** (checkable facts about the world that mark progress, such as "the deprecated API is gone from this file"
or "the ticket's status is resolved"), orders them into **stages**, and runs a **tracker** that says, for each step,
where the work stands. The tracker's main output is **Φ**, the weighted share of milestones reached. M6 turns the
change in Φ at each step into step-level credit, and the final Φ into bounded partial credit for failed attempts.

**Contract.**

| | |
|---|---|
| Inputs | `crucible.trace/v1` traces from M3 with signals from M4; a state view per step (a live World's `VerifierView`, a replayed `FileState`, or a `LogStateView`); outcome labels |
| Outputs | `Step` segmentation (`workflow/steps.py`); per-step state; mined `Milestone`s with filter verdicts; `StageModel`; `WorkflowGraph` (`crucible.workflow/v1`, `workflow/graph.py`); per-step `TrackerState` and Φ (`workflow/tracker.py`, `workflow/potential.py`) |
| Called by | pack `credit` hooks (toy `packs/_toy/.../studio.py:14-35`, coding `packs/coding/.../studio.py:147-175`), `crucible model` (`cli/studio.py:593`), M6 `defect_to_check`, M8 span derivation (`packs/coding/.../forge/derive.py:264`) |
| Layer | `workflow` sits above `connectors | signals`, below `credit` (ARCHITECTURE §1.5) |
| Principles | P12 credit goes to states, not actions; P13 shaping vs terminal partial credit; P16 infra failures are masked, not scored; P27 probes recorded, exposures mask; P28 "credit follows writes" |

**Gate status** (`docs/modules/M05-workflow.md`): S1 replay passes (91.9 % of 913 OpenHands attempts; 24/24 toy).
Tracker κ passes only at the pack's monotone threshold 0.6 (κ 0.87; 0.28 at the module default). Rediscovery is 4 of 6
check families. S2 (Φ at the midpoint predicts the outcome), shared with M6, fails: AUC 0.50 vs an LLM judge's 0.75;
a revision of S2 is pending in M6 (`docs/modules/M06-credit.md` findings 1–3).

**Where M5's output reaches a customer.** 
Established by tracing every importer of `crucible.workflow` (grep over `src/` and `packs/`) and every reader of the
per-step rows.

| Consumer | Uses M5's mined model? | Evidence |
|---|---|---|
| Live RL episodes (M10 runtime → M12 trainer) | **No.** Φ comes from the task's grader subgoals, tracked online; per-call credit is filtered to the policy's own rows | `runtime/harness/native.py:319` (`ProgressTracker(attempt.task.subgoals)`), `verify/progress.py:79-160`, `runtime/record.py:29-40` |
| `crucible label` (M15): per-turn `weight` in SFT / KTO / chain rows, `--mask-negative`, DPO turn evidence, `reward` in `rewards.jsonl` | **Yes**, through the pack's `credit` hook (`phi_by_seq`, composed reward) | `studio/label.py:171-184`; `studio/datasets.py:557-616` (turn weight = Φ after the turn − Φ before it), `:789`, `:842-843` |
| M8 span-task derivation (coding) | Yes, mined diff milestones as span checks | `packs/coding/.../forge/derive.py:264-265`; assessed in the M8 report |
| M6 `defect_to_check` | Yes, diff milestones as defect checks | `credit/operators/defect_to_check.py`; assessed in the M6 report |

So the M5 output that a customer trains on is the `label` path, and that is where R1 (§7) applies.

---

## 2. Status ledger

Sources: the 10-11 study (`docs/research/improvements/M03-M05.md`, M5 rows 1–8), the 10-12 sweep picks
(`docs/audit/2026-10-12/M05-M07-study-sweep.md:20-35`), and the open items of the M5 audit
(`docs/audit/2026-10-12/M05-workflow.md` §5).

| Item | State | Evidence |
|---|---|---|
| 10-11 #1 Fix and wire `from_signals` | **partly**: fixed, never called | fixed by audit W12 (`workflow/milestones/mine.py:224-248`); no caller in `src/`, `packs/` or `scripts/` |
| 10-11 #2 Automated nop/oracle Φ control per family graph | **not done** | no `milestones/controls.py`; pack `fixtures` hooks test the task grader's reward, not the mined graph's Φ (`packs/_toy/.../studio.py:173-188`) |
| 10-11 #3 Minefield/guardrail milestones | **not done** | no polarity or guardrail field in `core/schema/milestone.py:38-48`; base-true candidates are discarded (`workflow/milestones/filters.py:38-39`) |
| 10-11 #4 DFS optimal milestone mapping | **superseded** | with binary φ and a transitive order constraint, earliest-feasible assignment is already optimal (an exchange argument: moving any milestone later never admits one that was refused). ToolSandbox needs DFS because its similarities are soft (`NVIDIA-NeMo-Gym/.../evaluation.py:1196-1203@4fbffa1fe`) |
| 10-11 #5 `workflow/fidelity.py` (replay, κ, rediscovery, next-milestone accuracy) | **not done** | file absent although ARCHITECTURE §4 M5 lists it; `TrackerState.next_expected` is consumed nowhere |
| 10-11 #6 Bit-exactness from reference-differential CI | **not done** | rediscovery 4/6 (`docs/modules/M05-workflow.md`) |
| 10-11 #7 Sandbox re-execution fallback | **not done** | not recommended now (§7, R12) |
| 10-11 #8 Edge support and n; flag low-support stages; coding stage labels | **partly** | `EdgeSpec` carries support and n (`workflow/graph.py:22-26`); no low-support flag; no coding labels |
| Sweep pick 1 Fail-closed stale paths | **done** | `workflow/filestate.py:38`, `replay.py:138-161` |
| Sweep pick 2 Per-step state key | **half-built** | `Step.state_ref` is copied from events (`workflow/steps.py:40,119`), but no World, runtime or actor code writes one, and nothing reads it (grep over `src/`) |
| Sweep pick 3 Per-call credit rows | **done** | `workflow/steps.py:86` |
| Sweep pick 4 Submitted-scope replay | **done** | `workflow/replay.py:94-98` |
| Sweep pick 5 CI mining fixes | **built, unused** | `mine.py:188-221`; `from_ci` has no caller |
| Sweep picks 6–12 | **none built** | re-judged in §5: 6 and 8 carry conditions; 9, 10 and 11 assessed in §7 (first-version R10, R9, R4); 12 rejected |
| Audit open: coding stage labels, coding F1 graph, behavioural milestones, S2 rows recomputed | **not done** | — |

---

## 3. Comparison questions

| # | Question | Axis |
|---|---|---|
| Q1 | Are progress checkpoints defined over state or actions; how is out-of-order (stubbed) progress refused; are prerequisites transitive; where do they come from? | Correctness |
| Q2 | How are checkpoints derived, and how are they validated as predictive of success? | Correctness, Evidence |
| Q3 | How are case variants, branch-conditional steps and alternative correct implementations handled? | Correctness |
| Q4 | How is per-step state obtained for recorded trajectories, and what happens when it cannot be? | Robustness |
| Q5 | What is the step schema and segmentation unit; are there per-step state keys; is the progress model versioned? | Data model |
| Q6 | How are progress signals protected from farming (shallow checkpoints, oscillation, stubs, text claims)? | Security |
| Q7 | How is per-step evaluation cost controlled at scale and over long horizons? | Scale |
| Q8 | What does an operator see about where an agent is in a workflow? | Operability |
| Q9 | How does a non-coding, multi-app domain add progress checks without core changes? | Extensibility |
| Q10 | What statistical evidence backs induced structure? | Evidence |

---

## 4. Repo triage

All 31 repos. "Primary": has a real equivalent of part of M5's job. "Secondary": one reusable pattern.

| Repo | Class | Reason | Commit | Licence |
|---|---|---|---|---|
| NVIDIA-NeMo-Gym | primary | ToolSandbox milestone DAG, guardrails, minefields, id-free row matching; AutomationBench base-state roles; ATIF fail-closed projection | 4fbffa1fe | Apache-2.0 (ToolSandbox: Apple licence, pattern only) |
| ServiceNow-EnterpriseOps-Gym | primary | named per-workflow-step SQL state checks routed per app database | 271f2c3 | Apache-2.0 |
| PrimeIntellect-ai-prime-envs | primary | EnterpriseOps checks evaluated after every tool call; F2P/P2P triage | 91d7d15f6 | Apache-2.0 |
| PrimeIntellect-ai-community-environments | primary | native progress sources and the widest set of progress-farming counterexamples | cf220ee | Apache-2.0 |
| scaleapi-agentenv-framework | primary | per-tool-call changelog capture; generic state-trigger predicates over MCP apps; cursor live trajectories | 015c89b | Apache-2.0 |
| harbor-framework-harbor | primary | multi-step gating; ATIF; prefix seeding; live trajectory sync | 2821487d | Apache-2.0 |
| adithya-s-k-FineEnvs | primary | PortSim: observation-equality re-execution of recorded runs; decision-time attribution of exogenous change | 95235ba | Apache-2.0 (PortSim data CC BY-SA 4.0 / CC BY 4.0) |
| THUDM-slime | secondary | content-addressed workspace snapshot via a temporary git index | 05e3b017 | Apache-2.0 |
| alibaba-ROLL | secondary | per-step state hash and anchor-state groups | d96d0cce | Apache-2.0 |
| XiaomiMiMo-verl (+ mimoagent 467f0a1) | secondary | post-state database bundles; graded end-state levels; turn-depth horizon | a2ad9f61 | Apache-2.0 (mimoagent MIT) |
| areal-project-AReaL | secondary | one episode across compaction-split chains, refusing ambiguity | ba237a1b | Apache-2.0 |
| PrimeIntellect-ai-prime-agent | secondary | gate re-run only when the worktree fingerprint changed; golden-corpus normalisation | 33da15a2c | MIT |
| PrimeIntellect-ai-verifiers | secondary | `sampled` flag per message node; offline replay that skips runtime signals | 6d788c14a | MIT |
| huggingface-repo2rlenv | secondary | nop / oracle / valid-alternative / wrong-solution controls; cumulative test-defined steps | 27d04ee | Apache-2.0 |
| SWE-bench-SWE-smith | secondary | F2P/P2P derivation and gold-rerun repair | 9b74ac0 | MIT |
| SWE-agent-mini-swe-agent | secondary | trajectory format; records output, no state | 04d809ce | MIT |
| huggingface-OpenEnv | secondary | no progress mechanism; RolloutGraph discarded nodes; quick-submit hacking evidence | 9bcfa292 | BSD-3-Clause |
| NovaSky-AI-SkyRL | secondary | message graph with branch points (an M4 concern) | a1e34dbf | Apache-2.0 |
| FineEnvs-simulation-rl-environments | secondary | an exact mirror of FineEnvs' article; its claims rest on FineEnvs code | 9af60c3 | CC BY 4.0 |
| NVIDIA-NeMo-RL | not relevant | no segmentation, milestone or state code | f7f14b539 | Apache-2.0 |
| verl-project-verl | not relevant | `turn_scores` declared and never filled | 2bc6ac46 | Apache-2.0 |
| rllm-org-rllm | not relevant | step schema and reward differencing only | 4e828c81 | Apache-2.0 |
| OpenRLHF-OpenRLHF | not relevant | sums step rewards into one scalar | dc2a7ad | Apache-2.0 |
| huggingface-trl | not relevant | its environments are outcome-only | 7e15ebf5 | Apache-2.0 |
| MoonshotAI-checkpoint-engine | not relevant | weight transport only | 84631a2 | MIT |
| PrimeIntellect-ai-prime-rl | not relevant | one scalar per trace | 9b1904e7d | Apache-2.0 |
| preferencemodel-karotte | not relevant | an ordered step loop stopped by the judge; no state model | cbfc6bc | MIT |
| FineEnvs-multi-harness-rl | not relevant | terminal correctness plus a tool-count bonus | b780039 | CC BY 4.0 |
| AdithyaSK-rl-environments-guide | not relevant | prose only | ead51a4 | CC BY 4.0 |
| veds12-git-swarm | not relevant | a static site; its progress protocol is figure text | 9f66af0 | no licence |
| karpathy-autoresearch | not relevant | keep/revert over commits; no state predicates | 228791f | MIT |

The ten trainers were re-checked at today's commits: none has a workflow model, milestones or a validity gate,
confirming the 10-12 sweep.

---

## 5. Findings per repo

### NeMo-Gym (`4fbffa1fe`; +197 commits since the sweep's `3ef478d`)
- **ToolSandbox** (pattern only, Apple licence). A milestone is a set of constraints over per-app database snapshots
  (`resources_servers/toolsandbox/tool_sandbox/common/evaluation.py:726-797`). Matching walks a topological order and
  assigns each milestone a snapshot index no earlier than the last one assigned (`:1128-1129`), so order is transitive
  by construction. **Guardrails** require every database a milestone does not constrain to equal the reference
  snapshot (`:648-667,1019-1064`); **minefields** nullify the trajectory on any match (`:800-811,1004`). Rows are
  matched by attribute similarity with a Hungarian assignment, not by primary key (`:302-363`), and additions,
  removals and updates are judged as diffs (`:365-503`). Weaknesses: an empty milestone set scores 1.0
  (`:1274-1277`); conversation text is scored by ROUGE-L (`:671-680`). The matcher is not unit-tested in NeMo-Gym.
- **AutomationBench** (new since the sweep, commit `8908cb24d`). Each upstream assertion is evaluated on the initial
  world: true at base makes it a **guardrail**, false makes it an **objective**
  (`benchmarks/automationbench/automationbench_env/__init__.py:136-142`). A broken guardrail zeroes the task; the score
  is otherwise the share of objectives met (`:41-79`). Tested (`tests/test_assertion_results.py:33,59,69,80`).
- **workplace_assistant** re-executes recorded calls in a fresh multi-app environment (`resources_servers/workplace_assistant/utils.py:113-130`)
  but swallows any call that raises (`:127-129`): it guesses rather than failing closed.
- **ATIF projection** refuses what it cannot represent (`nemo_gym/atif_reverification.py:448-488`), confirmed unchanged.
- Verdicts. *Better than Crucible:* guardrail roles, transitive order, id-free row diffs. *Crucible better:* per-step Φ
  and a tracker (ToolSandbox pays one terminal similarity, `toolsandbox/app.py:346`); unmeasured stays unmeasured.
  *Mechanisms of interest:* transitive order, id-free rows, base-state roles (§7: not recommended now). *Reject:* empty-set = 1.0; text similarity
  as progress; exhaustive DFS as the default path.

### EnterpriseOps-Gym (`271f2c3`, unchanged)
- Checks are named SQL predicates over each app's database (`benchmark/verifier.py:81-127,374-399`), routed by
  `gym_name` (`verifier.py:42-46`, `executor.py:428-441`). The names are workflow step labels (`add_location`,
  `create_new_case`, `assign_case_to_user`). A new domain needs an MCP server, a seed SQL file and task JSON; no core
  change (Q9). Checks run once, at the end (`executor.py:360-363`); there is no order and no per-step evaluation.
- Defects found by probes against its shipped data: **9 of 98** `database_state` checks are already true on the
  untouched seed database; **23 of 102** checks are silently lost because results are keyed by name and later checks
  overwrite earlier ones (`executor.py:420,474`); `all([])` is true when every check is skipped (`:443-450,371`); a
  missing seed fails open (`benchmark/mcp_client.py:31-33`). Expected values pin auto-increment primary keys
  (`location_id = 907`), so a correct workflow that creates an extra row first fails. There are no tests.
- Verdicts. *Better:* an authored, multi-app, per-step state check set over real ITSM/CSM/HR tasks. *Crucible better:*
  duplicate milestone ids are rejected (`core/schema/milestone.py:59-61`); base-true candidates are filtered
  (`filters.py:38-39`). *Mechanism of interest:* authored checks as milestones (§7, first-version R9). *Reject:* name-keyed results,
  `all()` over an empty set, action-presence (`tool_execution`) and text (`response_check`) checks, fail-open seeding.

### prime-envs (`91d7d15f6`; +15)
- `enterprise_ops_gym` runs the task's SQL predicates at setup and **after every forwarded tool call**
  (`environments/tool_use/enterprise_ops_gym/enterprise_ops_gym/toolset.py:207,344`). An HTTP or SQL error scores
  `passed: False` (`:384-402`): fail-to-false. Each call overwrites the previous results (`models.py:62`), so no series
  is kept; the setup-time evaluation is computed and discarded. The pass fraction is a weight-0 metric.
- `pi_deepswe` records `triage = (f2p == 1 and reward == 0)`, i.e. "fixed the target, broke something that passed"
  (`verifier.py:420-446`): direct evidence for guardrail milestones.
- Verdicts. *Better:* per-call predicate evaluation over a database. *Crucible better:* keeps the series, tiers cost,
  filters base-true checks, keeps errors unmeasured. *Mechanism of interest:* per-call database predicates (§7, first-version R9). *Reject:* re-running every predicate after every call;
  fail-to-false.

### community-environments (`cf220ee`, unchanged)
- Native progress exists where the app has it (`nyt_connections.py:209-212` found groups / 4; `androidworld.py:113-116`
  the app's own success check). text_quests ships 327 authored checkpoints, each with a written rationale
  (`core/game_progress.json`), matched as substrings of observation text, kept as a high-water mark that survives a
  restore (`core/textquests_core.py:295,634-650`).
- Progress-farming shapes, each paid by its env: dwell sums (`balrog_bench.py:302-314`), count sums (`:279-287`) with a
  reasoning-length bonus (`:268-269`), a score read from model text (`hanabi/hanabi.py:389-408`; a probe shows an
  assistant message lifts 3 to 25), recall without precision (`browsecomp_plus.py:265-283`), self-written tests
  (`rust_cargo.py:76-129`). No env is tested on its reward.
- **Corrections to the sweep:** pick 10's source (`text_quests.py:402-408`) is text matching with a high-water mark,
  not App state; the right source is prime-envs' database predicates. Pick 12's information gain
  (`mastermind.py:347-376`) is a terminal scalar over a finite hypothesis space that no Crucible World has: rejected.
  scicode is not "Φ parity": earlier code is prefixed whether or not the earlier step passed (`scicode.py:349-355`).
  balrog's level counters are counterexamples, not sources.

### agentenv (`015c89b`; +62)
- **State triggers**: a generic predicate language over any app's MCP read tools (`extract` / `bind` / `predicate`),
  re-evaluated after every non-read-only call by a watched role (`src/agent_env/env/gateway/triggers.py:117-160,347-381,841-845`).
  Errors evaluate to false (`:1072-1104`). Tested (`tst/unit/env/gateway/triggers_test.py` and siblings). Crucible
  already compiles World `when.state` checks into this language (`interop/agentenv/world_backend.py:394-418`), so the
  World has a state-predicate language that milestones do not use.
- **Changelog**: per-tool-call workspace increments keyed by tool-call position, applied with a rewind cutoff
  (`src/agent_env/a2a_agent/object_transfer.py:505-592`). Capture is an optional agent feature
  (`packages/agentenv-protocol/.../extensions.py:1076-1095`) and no in-tree agent produces it; the validator checks
  conformance only (`validator.py:228-237`). **SnapshotSeries** is per simulated-user turn, capped at 24
  (`task_step/snapshot_utils/snapshot_series.py:56`), and graded by nothing.
- **New since the sweep** (`2bfb3cd`): a cursor-based live trajectory (`a2a_agent/trajectory_follower.py:34-123`) with
  explicit state and incremental pages.
- **Corrections to the sweep:** pick 6's "exact per-tool-call state" depends on an agent that implements changelog
  capture; pick 8 is better served by this cursor than by Harbor's file sync.
- Verdicts. *Better:* per-call capture defined in the protocol; incremental live trajectories; one generic predicate
  language. *Crucible better:* a progress concept at all; unmeasured instead of false. *Mechanisms of interest:* per-call state keys and
  one predicate language (§7: not recommended now). *Reject:* fail-to-false trigger semantics; SnapshotSeries as per-step state.

### Harbor (`2821487d`; +48)
- Multi-step tasks gate each step on the previous one's `min_reward` (`src/harbor/trial/multi_step.py:74-87,196-232`):
  positional, transitive, authored. ATIF v1.8 steps carry no state key (`src/harbor/models/trajectories/step.py:14-91`).
  rewardkit's `sqlite_query_equals` returns false when the database is missing or the query errors
  (`packages/rewardkit/src/rewardkit/criteria/sqlite_query_equals.py:14-26`). Commit `d5ac1be1` documents a stub with
  correctness 0.3194 scoring 1.0 when partial credit fed an any-pass aggregator.
- Prefix seeding (`src/harbor/trial/trial.py:1192-1283`) restores conversation only, never sandbox files, and only three
  installed agents load ATIF. Live sync (`src/harbor/trial/sync_trajectory.py:110-158`) is off by default
  (`models/trial/config.py:202`), disabled with a simulated user (`trial.py:649`), and re-converts the whole file every
  2 s.
- Verdicts. *Crucible better:* mined milestones, Φ, fail-closed state. *Adopt:* pick 7 (prefix seeding) belongs to
  M8/M11 export, not M5. *Reject:* Harbor live sync as the primary online feed.

### FineEnvs (`95235ba`; PortSim live added in `86f94b2`, after the sweep)
- Recorded runs are re-executed against a fresh deterministic environment and **every tool output is asserted
  byte-equal to the recording**, failing at the first divergence
  (`07-simulation-environments/portsim-live/envs/port_week/agentenv/tests/recorded.py:147-203`); tested on six recorded
  weeks (`test_recorded_runs.py:31-36,70-76`).
- Change that the agent did not cause is excused by what was known when the decision was made
  (`core/port_week_core/world.py:66-87`, `wind.py:232-273`; tested in `core/tests/test_world.py:150`).
- No milestones or Φ. *Mechanisms of interest:* attribution of exogenous change, verified re-execution (§7). *Correction to the sweep:* the cited
  `harbor.py:346-369` is the explorer catalog's prose renderer, not an executor.

### Secondary patterns
- **slime** (`05e3b017`, new `58b08786`): workspace snapshot as a git tree built under a temporary index, leaving the
  agent's index and HEAD untouched, validated as a hex object id (`examples/coding_agent_rl/swe.py:237-255`); tested
  for unstaged, staged and committed edits. Crucible has the same plumbing (`forge/repo.py:155-168`).
- **ROLL** (`d96d0cce`): `state_hash` is an exact SHA-256 of the observation (`roll/pipeline/agentic/env_manager/step_env_manager.py:63`),
  of the full message history for tool-calling agents (`agent_native_env_manager.py:200,221`), and of the step index
  for Harbor-backed ones (`proxy_env_manager.py:460`). Only the first is a state key.
- **MiMo fork**: opt-in post-state bundles of the workspace and each app's database
  (`recipes/general/general_agent/environment.py:620-670`).
- **prime-agent** (`33da15a2c`): a failed gate is re-run only when the worktree fingerprint changed; a failed capture
  forces a re-run (`crates/.../autonomous/gates.rs:28-38,145-229,277-317`); tested. `golden_replay.rs` is a test-only
  parity check of 69 single-call fixtures, not a trace replayer (corrects the 10-11 study's #7 citation); its
  volatile-token normalisation (`:33-58`) is reusable as a validation harness.
- **repo2rlenv** (`27d04ee`): controls with nop, oracle, a valid alternative and a wrong solution
  (`src/repo2rlenv/quality/loop/runner.py:76-111`), tested.

---

## 6. Gap matrix

`●` built and tested, `◐` built with a stated weakness, `○` absent.

| Q | Crucible today | ToolSandbox / AutomationBench | EnterpriseOps-Gym / prime-envs | agentenv | FineEnvs PortSim | slime / ROLL |
|---|---|---|---|---|---|---|
| Q1 state-based, transitive order | ◐ state-based; order not transitive in Φ; prerequisites never set | ● order by snapshot index | ○ unordered | ○ | ○ irreversibility only | ○ |
| Q2 validated as predictive | ○ filter (d) unapplied | ○ | ○ (9/98 base-true) | ○ | ◐ graded against an optimum | ○ |
| Q3 variants, alternatives | ○ frequency rule drops variants; token milestones | ● id-free row diffs | ◐ per-task by authoring; pinned ids | ○ | ● outcome-only | ○ |
| Q4 per-step state for records | ◐ text replay, fail-closed | ◐ captured at record time | ○ end only | ◐ changelog (agent-provided) | ● verified re-execution | ● git-tree capture (start/end) |
| Q5 state keys, versioned model | ◐ field exists, never produced; model not persisted | ○ | ○ | ● position-keyed | ◐ reward version per task | ◐ exact hash |
| Q6 anti-farming | ◐ monotone filter, non-trivial; no guardrails; other actors' writes credited | ● guardrails, minefields | ◐ prime-envs triage | ○ | ● decision-time excuses | ○ |
| Q7 evaluation cost | ◐ cost tiers, stale carry-forward | ◐ cached per (milestone, snapshot) | ○ all checks every call | ● only after mutating calls | ○ | ○ |
| Q8 operator view | ◐ tracker offline only | ◐ per-assertion records | ◐ per-check pass rates | ● trigger log, live pages | ● audit data | ◐ group-size metric |
| Q9 new domains without core change | ○ core miner coding-only | ● per-app snapshots | ● routed per app | ● generic predicates | ◐ subclass hooks | ● observation as key |
| Q10 statistics on structure | ◐ thresholds without confidence | ○ | ○ | ○ | ○ | ○ |

---

## 7. Recommendations, ranked

Only recommendations that pass the bar of the internal audit plan's Step 7 (`00-plan.md`, gitignored) are listed: the change fixes something wrong in what
Crucible produces today, its benefit is measured, and it cannot break a later stage. One passes. The candidates that
did not pass follow the recommendation, each with its reason.

### R1. Φ reaches a dataset only through a held-out validity verdict, one per channel
- **Kind** defect (against DESIGN §5.3 filter (d) and §6.5) · **Status** new; replaces the first version's R4, R6 and
  R13.
- **Problem.**
  - *No gate on the product path.* DESIGN §6.5 keeps a progress signal only if it predicts recorded outcomes on
    held-out, later traces (AUC ≥ 0.6). Crucible implements the check as `credit.validity.validate`, with a time split
    and a group bootstrap (`credit/validity.py:59,97,133`). Its only product caller is the rubric-item operator
    (`credit/operators/rubric_item.py:103`).
  - *Both credit hooks emit Φ unconditionally.* The toy hook emits `phi_by_seq` and a reward with partial credit for
    every trace (`packs/_toy/.../studio.py:30-33`, `m56/pipeline.py:237`). The coding hook does the same
    (`packs/coding/.../studio.py:170-171`).
  - *S2 lives in a script.* It is computed by `packs/coding/.../m56/rebench_s2.py`, so its result never changes what
    `label` writes.
  - **Toy:** two of the three mined milestones (`log/assignee`, `log/resolution`) are present at the end of every
    ticket, good or bad (held-out AUC exactly 0.50). `assignee` is written by the queue system in every ticket.
    - Φ at the end is 1.0 in 4 of the 6 bad tickets.
    - Φ at the midpoint, from which per-turn weights are built, has AUC 0.40 on the dev tickets and 0.50 (corrected from 0.58) on the
      held-out tickets.
    - What makes a ticket bad (a wrong answer, a low CSAT score) is in no end-state field.
    - In `kto.jsonl`, the wrong answer of failed ticket T1011 ("Your billing country is US, so prices are in USD.")
      carries weight **+0.667**, the highest in the file. Correct answers of good tickets carry 0.0.
    - In `rewards.jsonl`, T1011 and T1019 (wrong answers, reopened by the customer) earn partial credit 0.133.
  - **Coding:** Φ at the midpoint has AUC 0.50 (CI 0.42–0.58) on 200 held-out SWE-rebench attempts. End-state Φ is
    informative: AUC 0.66 (CI 0.53–0.80), within-issue 0.72 (`docs/modules/reports/m6_s2_swe_rebench.json`).
    - The hook places Φ_end on the trace's last event, and `from_trace` gives it all to the final policy turn, or to
      no turn when the last event is not the policy's.
    - Every row is still flagged `credit: true`.
- **Reference.** None. No repo among the 31 validates progress checkpoints against outcomes before paying for them. The
  10-12 sweep and this study's five repo groups found the same (Q2, Q10 in §6). The rule is DESIGN's own, and the code
  is Crucible's own.
- **Recommendation.**
  - *Two channels, two verdicts.* For each family's workflow model, `label` computes one verdict per channel on
    held-out traces.
    - **Step channel**: per-turn weights, the ΔΦ used by `--mask-negative`, and the DPO turn evidence. It is released
      when Φ at intermediate states predicts the outcome. The statistic is the one M6's gate S2 defines (today, Φ at
      the midpoint).
    - **Terminal channel**: the partial-credit term κ(1−O)Φ of the reward. It is released when Φ at the end state
      predicts the outcome.
  - *Decision rule.* Held-out traces come from `time_split` (latest 30 %), with the bootstrap grouped by task. A channel
    is released when the held-out AUC is at least 0.6 **and** the 95 % lower confidence bound is above 0.5.
  - *Fallback.* An unreleased channel uses the existing "no dense credit" encoding.
    - Step weights are null (`studio/datasets.py:563`); rows carry 0.0 with `credit: false`.
    - The reward is composed with Φ = 0, i.e. outcome-only (`credit/compose.py:71`).
    - A family without both outcome classes is unreleased with the reason `holdout lacks one class`.
  - *Visibility.* Each channel's verdict (AUC, CI, traces per class, reason) is written to the label run's
    `summary.json` and named in every reward row. Each milestone's validity is reported beside it.
  - *No per-milestone weighting.* Per-milestone AUCs are reported but not used as weights.
- **Reasoning.**
  - *The lower bound is required.* The point rule alone passes the toy's end-state Φ at 0.667 on three failures, while
    its interval is [0.50, 1.00].
  - *Per-milestone weighting is excluded.* At tens of traces the per-milestone AUCs are noise (the toy's
    `status=resolved`: 0.667, CI [0.50, 1.00]). On SWE-rebench, weighting did not move S2 (midpoint AUC 0.499 weighted
    vs 0.490 uniform).
  - *The fallback cannot break a later stage.* It is the path Crucible already takes when no Φ exists.
    - No trainer in Crucible reads the weight column (grep over `src/crucible/train/`).
    - A customer's trainer that does read it no longer receives numbers that invert the outcome.
  - *What works is kept.* Coding keeps its end-state partial credit; only the per-turn placement, which was never
    credit, goes.
  - *The S2 decision governs the data.* Whatever M6 decides for S2 changes what customers receive.
  - *Two first-version cases are covered without separate fixes:*
    - system-written and every-case milestones carry no outcome information, so they cannot open a channel;
    - chance-level per-step Φ on coding stays blocked until S2 passes.
  - *Trade-off accepted:* dense credit needs labelled failures. A signal of the toy's strength (a third of failures
    separable) clears the lower bound at 12 held-out failures. A continuous signal at AUC 0.66 needs about 80 traces
    per class. A customer with only successful logs gets outcome-only credit and the stated reason; no measurement
    shows Φ helps such a customer.
- **Established by.**
  - **[ran: `m5v/v1`, `v2`, `v3`]** the per-ticket Φ, rewards, turn weights and writers; the end-state fields by
    outcome; `validate` on dev and held-out tickets.
  - **[ran: two `crucible label --pack toy` runs, with and without the gate (`v4`)]** identical row sets (10 SFT,
    12 KTO, 0 DPO, same tasks and steps). The only differences were weights 0.0 with `credit: false`, and the T1011 /
    T1019 rewards going from 0.133 to 0.
  - **[ran: `v5`]** where an end-state Φ lands as a turn weight.
  - **[ran: `v6`]** how many labelled failures the gate needs.
  - **Read:** the S2 report (116 resolved / 84 failed held-out attempts), and every importer of `crucible.workflow`
    and every reader of the per-step rows (§1).
- **Impact** H: stops uninformative and inverted numbers from entering customer training data. **Effort** S:
  `validate`, `time_split` and `bootstrap_ci` exist, and the hooks' outputs are already optional. **Risk** L: the
  fallback is an existing path, validated above.
- **Mode** — (Crucible's own design and code).
- **Acceptance.**
  - **Toy:** the label summary reports both channels unreleased (step: held-out AUC 0.50 < 0.6; terminal: lower bound
    0.50). The SFT / KTO / DPO row sets equal today's, every row says `credit: false`, and the T1011 / T1019 rewards
    are 0.
  - **Coding:** on the S2 SWE-rebench data the terminal channel is released (AUC 0.66, lower bound 0.53) and the step
    channel is not (0.50). The coding pack's own `label` run is measured once the stale `pyproject.toml` sources are
    fixed (§9, §10).
  - **Unit tests:**
    - a synthetic family whose milestone separates 20 good from 20 bad traces releases both channels;
    - a family without failures is unreleased with `holdout lacks one class`;
    - a family that passes the point rule but not the lower bound is unreleased.

### Candidates not recommended

Each item below was in the first version. It is not recommended now for the reason given. Items marked **premise
corrected** rested on a claim that the re-test disproved.

| First-version item | Why it does not meet the bar now |
|---|---|
| v1-R1 One transitive prerequisite rule for Φ, `credited` and the matcher | The disagreement is real **[ran: `m5_probe1`]**, but no product path sets `requires` on a milestone. The only one is a toy smoke test (`packs/_toy/.../smoke.py:59`), so no output changes today. It becomes a defect the day prerequisites are introduced; fix it then, together with that change. |
| v1-R2 Mined precedence becomes prerequisites | **Not safe.** On 8 traces a perfect order has a 95 % lower bound of 0.68. Turning observed order into a hard prerequisite would refuse valid orders that the sample did not contain. |
| v1-R3 Credit follows the writer | **Premise corrected.** In `label`, other actors' writes do not reach policy-turn weights on the toy: every non-policy event carries text, which closes the open turn first **[ran: `v1`]**. Live RL credits only the policy's own rows (`runtime/record.py:33-34`). What remains is partial credit that includes system writes; the recommended R1 withholds it, because such milestones carry no outcome information. A silent non-policy state event would still leak into a turn (§10, M15). |
| v1-R4 Per-milestone predictive filter and AUC weights | Merged into R1 as a per-channel gate. Per-milestone weighting dropped: noisy at tens of traces, and no measured gain on S2. |
| v1-R5 Stale values and bisection for late confirmations | Latent. No medium or expensive milestone exists in any product path (toy and coding milestones are cheap; `from_ci` has no caller). |
| v1-R6 Coding product path emits per-step Φ | **Premise corrected.** It would export Φ at the midpoint, which is at chance (AUC 0.50). Under R1 this stays blocked until S2 passes. |
| v1-R7 Versioned, persisted workflow model | The gap is real, but mining is deterministic for the same inputs, and nothing consumes a graph citation today. Not critical. |
| v1-R8 Contrastive, case-conditioned event-log miner | **Not validated.** In the toy, what separates bad tickets (a wrong answer, a low CSAT score) is not in the state at all **[ran: `v2`]**, and 12 dev tickets span 8 case types, one good trace each. No miner can be shown to help on the data available. |
| v1-R9 Import authored state predicates (EnterpriseOps-Gym) | An extension with no customer path consuming it today. Imported checks would also go through R1. |
| v1-R10 Guardrail milestones from base-true candidates | Changes the reward's meaning. "True at base" is not "must be preserved": fields legitimately change. Not validated on data. |
| v1-R11 Exact per-step `state_ref` at record time | **Premise corrected.** Crucible's own episodes compute Φ live from grader subgoals (`runtime/harness/native.py:319`), not by replaying call text. A state key would help only offline relabelling of external traces. |
| v1-R12 Verified re-execution of external traces | No reconstructable environment exists for the external traces at hand, so the benefit cannot be measured. |
| v1-R13 Confidence bounds instead of raw shares; fidelity report | The core (decide on a lower confidence bound) is part of R1's decision rule. The rest is reporting. |

---

## 8. What Crucible already does better (keep)

- **Milestones, stages and an online tracker exist at all.** No reference mines progress predicates, induces stages or
  locates an agent in a workflow; every reference pays one terminal score or an authored per-step test.
- **Unmeasured is not zero.** A check that cannot run yields `None` (`workflow/milestones/evaluate.py:64-66`). Harbor's
  rewardkit, agentenv triggers, prime-envs and EnterpriseOps-Gym all turn errors into false.
- **Replay fails closed per path and is versioned** (`workflow/filestate.py:38`, `replay.py:225`); workplace_assistant
  swallows failures.
- **Milestone sets reject duplicates and unknown prerequisites** (`core/schema/milestone.py:59-68`); EnterpriseOps-Gym
  silently loses 23 % of its checks to name collisions.
- **Base-true candidates are not paid as progress** (`filters.py:38-39`); EnterpriseOps-Gym ships 9 % of its checks true
  at base.
- **Shell-effect replay** of recorded commands has no counterpart in any reference.

---

## 9. Open questions

- **Coding verdicts on the product path.** The coding numbers in R1 come from the S2 report on SWE-rebench
  (`docs/modules/reports/m6_s2_swe_rebench.json`). The coding pack's own `label` run could not be executed: it imports
  Harbor, which cannot be installed from this checkout (§10). The verdict for the coding pack's recorded seed runs is
  measured once that is fixed. Until then it is not established.
- **Customer sample sizes.** R1 (§7) states the number of labelled failures the gate needs. How many labelled failures a
  typical customer workflow has is not known yet; it decides how often customers see dense credit.

---

## 10. Handoff: findings outside M5

| To | Finding |
|---|---|
| M0 / M11 / M14 | `crucible/pyproject.toml:98-101` points `harbor`, `verifiers` and `agentenv` at `../RL-Environments/06-frameworks-and-env-code/…` and `../agentenv-framework`, neither of which exists. The `harbor`, `verifiers` and `agentenv` extras cannot install; `crucible.interop.harbor.*`, the coding pack (`seeds.py:17`) and every agentenv dependency are unimportable. `crucible label --pack coding` fails on this checkout (`no connector named 'agent_runs'`: the connector imports Harbor) and also needs `data/cache/{github,git}`, which a clean checkout lacks. |
| M6 | S2 is pending a decision (`docs/modules/M06-credit.md` finding 3). Under R1's rule that decision sets the step-channel statistic, so it decides what customers receive. It is no longer only a report. |
| M6 | Toy reward vs outcome label: T1012 and T1018 are labelled bad (CSAT 1 and 2) but their reward is 1.0, because the toy reward's outcome is "resolved in state" (`packs/_toy/.../m56/pipeline.py:231-237`) **[ran: `v1`]**. `label` already drops these from KTO (`kto_outcome_disagrees 2`). The reward's outcome is assessed in the M6 report. |
| M6 | Add the farming shapes found in community-environments (dwell sums, count sums, a restore that keeps progress, scores read from text, recall without precision) and Harbor `d5ac1be1` (partial credit through an any-pass aggregator) as R-property fixtures in `credit/properties.py`. |
| M6 / M12 | `StepRow` has no flag for model-sampled vs copied context (`core/schema/reward.py:65-77`); verifiers carries `MessageNode.sampled` (`verifiers/v1/graph.py:103-106@6d788c14a`). Sweep G3 pick 1, still open. |
| M15 | `studio/datasets.py:557-616`: a non-policy `state` event *without text* inside an open policy turn adds its ΔΦ to that turn, because only events with text close the turn. It does not occur in the toy data (every event has text) **[ran: `v1`]**; connectors that emit silent state rows would trigger it. The turn boundary should close on any non-policy write. |

---

## 11. Sources

**Reference repos** (`RL-Environments-Frameworks/`): NVIDIA-NeMo-Gym `4fbffa1fe`, ServiceNow-EnterpriseOps-Gym `271f2c3`,
PrimeIntellect-ai-prime-envs `91d7d15f6`, PrimeIntellect-ai-community-environments `cf220ee`,
scaleapi-agentenv-framework `015c89b`, harbor-framework-harbor `2821487d`, adithya-s-k-FineEnvs `95235ba`, THUDM-slime
`05e3b017`, alibaba-ROLL `d96d0cce`, XiaomiMiMo-verl `a2ad9f61` (mimoagent `467f0a1`), areal-project-AReaL `ba237a1b`,
PrimeIntellect-ai-prime-agent `33da15a2c`, PrimeIntellect-ai-verifiers `6d788c14a`, huggingface-repo2rlenv `27d04ee`,
SWE-bench-SWE-smith `9b74ac0`, SWE-agent-mini-swe-agent `04d809ce`, huggingface-OpenEnv `9bcfa292`, NovaSky-AI-SkyRL
`a1e34dbf`, FineEnvs-simulation-rl-environments `9af60c3`, NVIDIA-NeMo-RL `f7f14b539`, verl-project-verl `2bc6ac46`,
rllm-org-rllm `4e828c81`, OpenRLHF-OpenRLHF `dc2a7ad`, huggingface-trl `7e15ebf5`, MoonshotAI-checkpoint-engine
`84631a2`, PrimeIntellect-ai-prime-rl `9b1904e7d`, preferencemodel-karotte `cbfc6bc`, FineEnvs-multi-harness-rl
`b780039`, AdithyaSK-rl-environments-guide `ead51a4`, veds12-git-swarm `9f66af0`, karpathy-autoresearch `228791f`.

**Crucible documents:** `DESIGN.md` §3, §5.3, §6.2, §6.4, §6.5; `ARCHITECTURE.md` §1.5, §3, §4 M5;
`docs/modules/M05-workflow.md`; `docs/audit/2026-10-12/M05-workflow.md`, `M05-study-defects.md`,
`M05-M07-study-sweep.md`; `docs/research/improvements/M03-M05.md`; `docs/modules/reports/m6_s2_swe_rebench.json`,
`m56_toy.json`.

**Crucible code:** `src/crucible/workflow/` (all files), `core/schema/milestone.py`, `core/schema/trace.py`,
`credit/operators/rubric_item.py`, `credit/validity.py`, `studio/label.py`, `studio/datasets.py`, `cli/studio.py`,
`world/world.py`, `forge/repo.py`, `interop/agentenv/world_backend.py`, `packs/_toy/src/crucible_pack_toy/{studio.py,m56/pipeline.py}`,
`packs/coding/src/crucible_pack_coding/{studio.py,m56/rebench_s2.py,forge/derive.py}`, `runtime/harness/native.py`,
`runtime/record.py`, `verify/progress.py`, `credit/compose.py`.

**Probes run for this report** (Crucible venv; no reference code executed except the EnterpriseOps-Gym executor
function under stubs and an AST extract of one hanabi function): `m5_probe1.py` (prerequisite semantics),
`m5_probe2.py` (toy graph), `m5_probe3.py` (toy field frequencies), `probe4_scale.py` (evaluation cost),
`probe5_delay.py` (credit timing), `g3probe/p1_stale_carry.py`, `g6/probe_actor.py`, `probes/eog_base_vacuity.py`,
`probes/eog_name_collision.py`, `g3probe/p2_hanabi_claim.py`.

**Probes added in the revision** (`scratchpad/m5v/`):
- `v1_label_effect.py`: per toy ticket, its outcome, end-state Φ, reward and turn weights, and which actor moved Φ.
- `v2_fields.py`: end-state fields by outcome, and the bad tickets' events.
- `v3_gate.py`: `credit.validity.validate` on the toy graph, with dev fit and held-out test, for end-state Φ,
  midpoint Φ and each milestone.
- `v4_gated_label.py`: `crucible label --pack toy` with the gate applied, diffed against the ungated run.
- `v5_endphi_turns.py`: where a single end-state Φ lands as a turn weight.
- `v6_samples.py`: how many labelled failures the lower bound needs.
